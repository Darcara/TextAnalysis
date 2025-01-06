namespace TextAnalysis;

using System.IO.Hashing;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Neco.Common;
using Neco.Common.Extensions;
using SentencePieceTokenizer;

public sealed class OnnxSession : IDisposable {
	public readonly SessionConfiguration Configuration;
	internal readonly SessionOptions Options;
	private readonly Boolean _disposeOptions;
	public readonly InferenceSession Session;
	public readonly RunOptions RunOptions;

	public OnnxSession(SessionConfiguration configuration, InferenceSession session, SessionOptions options, Boolean disposeOptions) {
		Configuration = configuration;
		Session = session;
		Options = options;
		_disposeOptions = disposeOptions;
		RunOptions = new RunOptions();
	}

	#region IDisposable

	/// <inheritdoc />
	public void Dispose() {
		RunOptions.Dispose();
		Session.Dispose();
		if (_disposeOptions)
			Options.Dispose();
	}

	#endregion
}

internal class OnnxModelLoader {
	private readonly ILogger _logger;

	public OnnxModelLoader(ILogger logger) {
		_logger = logger;
	}

	public OnnxSession Load(String modelPath, SessionConfiguration sessionConfiguration, SessionOptions? sharedOptions = null) {
		switch (sessionConfiguration.ExecutionProvider) {
			case ExecutionProvider.CPU:
				return LoadCpu(modelPath, sessionConfiguration, sharedOptions);
			case ExecutionProvider.DirectML:
				return LoadDml(modelPath, sessionConfiguration, sharedOptions);
			default:
				throw new ArgumentOutOfRangeException($"Unknown ExecutionProvider: {sessionConfiguration.ExecutionProvider}, must be one of {String.Join(", ", Enum.GetNames(typeof(ExecutionProvider)))}");
		}
	}

	internal static SessionOptions CreateSessionOptions(SessionConfiguration sessionConfiguration, String? optimizedOutput = null) {
		SessionOptions opt = new() {
			IntraOpNumThreads = sessionConfiguration.IntraOpNumThreads,
			InterOpNumThreads = sessionConfiguration.InterOpNumThreads,
			ExecutionMode = sessionConfiguration.InterOpNumThreads > 1 ? ExecutionMode.ORT_PARALLEL : ExecutionMode.ORT_SEQUENTIAL,
			GraphOptimizationLevel = sessionConfiguration.OptimizationLevel,
			OptimizedModelFilePath = optimizedOutput,
		};
		opt.RegisterOrtExtensions();

		if (sessionConfiguration.EnableVerboseOrtLogging) {
			opt.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
			opt.LogVerbosityLevel = 100;
		}
		// TODO LogLevel
		// TODO profiling

		// opt.DisablePerSessionThreads(); // requires: OrtEnv.CreateInstanceWithOptions();

		// seems to have problems with merged
		// but positive for size
		// unknown for speed
		// opt.AddFreeDimensionOverrideByName("batch_size", 1);
		// opt.AddFreeDimensionOverrideByName("encoder_sequence_length", 16);

		if (sessionConfiguration.FreeDimensionOverrides != null) {
			foreach (FreeDimensionOverride ovr in sessionConfiguration.FreeDimensionOverrides) {
				if (ovr.Type == DimensionOverrideType.ByName)
					opt.AddFreeDimensionOverrideByName(ovr.Key, ovr.Value);
				else if (ovr.Type == DimensionOverrideType.ByDenotation)
					opt.AddFreeDimensionOverride(ovr.Key, ovr.Value);
				else
					throw new InvalidOperationException($"Invalid free dimension override type: {ovr}");
			}
		}
		
		if (!String.IsNullOrEmpty(sessionConfiguration.Batching.BatchingDimensionName))
			opt.AddFreeDimensionOverrideByName(sessionConfiguration.Batching.BatchingDimensionName, sessionConfiguration.Batching.BatchSize);


		switch (sessionConfiguration.ExecutionProvider) {
			case ExecutionProvider.CPU:
				opt.AppendExecutionProvider_CPU();
				break;

			case ExecutionProvider.DirectML:
				try {
					// https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html#configuration-options
					opt.EnableMemoryPattern = false;
					opt.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
					opt.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
					opt.AppendExecutionProvider_DML(sessionConfiguration.GpuDeviceId);
					// CPU fallback has to be defined AFTER DML endpoint
					// CPU endpoint is always implicitly available, but may generate ORT warnings during model load, such as
					// "VerifyEachNodeIsAssignedToAnEp": Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf. 
					opt.AppendExecutionProvider_CPU();
				}
				catch (Exception e) {
					throw new ArgumentException($"Execution provider '{sessionConfiguration.ExecutionProvider}' unavailable. Are you missing the proper ONNX runtime? Available execution providers: {String.Join(", ", OrtEnv.Instance().GetAvailableProviders())}", e);
				}

				break;
			default:
				throw new ArgumentOutOfRangeException($"Unknown ExecutionProvider: {sessionConfiguration.ExecutionProvider}, must be one of {String.Join(", ", Enum.GetNames(typeof(ExecutionProvider)))}");
		}


		if (sessionConfiguration.EnableGeluApproximation) opt.AddSessionConfigEntry("optimization.enable_gelu_approximation", "1");
		if (sessionConfiguration.EnableGemmFastMath) opt.AddSessionConfigEntry("mlas.enable_gemm_fastmath_arm64_bfloat16", "1");
		if (sessionConfiguration.DisableAheadOfTimeFunctionInlining) opt.AddSessionConfigEntry("session.disable_aot_function_inlining", "1");
		if (sessionConfiguration.UseDeviceAllocatorForInitializers) opt.AddSessionConfigEntry("session.use_device_allocator_for_initializers", "1");
		if (sessionConfiguration.IntraOpThreadAffinities != null) opt.AddSessionConfigEntry("session.intra_op_thread_affinities", sessionConfiguration.IntraOpThreadAffinities);

		return opt;
	}

	private static String GetModelName(String baseName, SessionConfiguration sessionConfiguration) {
		if (sessionConfiguration.OptimizationLevel == GraphOptimizationLevel.ORT_DISABLE_ALL)
			return baseName;

		StringBuilder keyBuilder = new();
		if (sessionConfiguration.FreeDimensionOverrides != null) {
			foreach (FreeDimensionOverride ovr in sessionConfiguration.FreeDimensionOverrides) {
				keyBuilder.AppendLine($"FreeDimensionOverride: {ovr}");
			}
		}

		if (!String.IsNullOrEmpty(sessionConfiguration.Batching.BatchingDimensionName))
			keyBuilder.AppendLine($"Batching: {sessionConfiguration.Batching.BatchingDimensionName}={sessionConfiguration.Batching.BatchSize}");
			
		keyBuilder.AppendLine($"InterOp: {sessionConfiguration.InterOpNumThreads}");
		keyBuilder.AppendLine($"IntraOp: {sessionConfiguration.IntraOpNumThreads}");

		String key = String.Empty;
		if (keyBuilder.Length > 0)
			key = XxHash3.HashToUInt64(MagicNumbers.Utf8NoBom.GetBytes(keyBuilder.ToString())).ToString("X16");

		String readablePart = sessionConfiguration.ExecutionProvider switch {
			ExecutionProvider.DirectML => "DML",
			_ => sessionConfiguration.OptimizationLevel.ToString(),
		};

		return Path.ChangeExtension(baseName, $"{key}.{readablePart}.onnx");
	}

	private String OptimizeIfNecessary(String modelBasename, SessionConfiguration sessionConfiguration) {
		String targetModelName = GetModelName(modelBasename, sessionConfiguration);

		using SessionOptions sessionOptions = CreateSessionOptions(sessionConfiguration, targetModelName);
		using InferenceSession localModelForOptimization = new(modelBasename, sessionOptions);

		Int64 originalSize = new FileInfo(modelBasename).Length;
		Int64 optimizedSize = new FileInfo(sessionOptions.OptimizedModelFilePath).Length;
		_logger.LogDebug("Optimized {BaseModelName} ({BaseModelSize}) to {OptimizedModelName} ({OptimizedModelSize}) = {SgnSizeDelta}{SizeDelta} or {SgnSizeDeltaPct}{SizeDeltaPct:P2}", Path.GetFileName(modelBasename), originalSize.ToFileSize(), Path.GetFileName(sessionOptions.OptimizedModelFilePath), optimizedSize.ToFileSize(), (optimizedSize > originalSize ? "+" : ""), (optimizedSize - originalSize).ToFileSize(), (optimizedSize > originalSize ? "+" : ""), (optimizedSize - originalSize) / (Double)originalSize);

		return targetModelName;
	}

	private String FindPath(String baseModelPath, SessionConfiguration? sessionConfiguration = null) {
		String modelName = GetModelName(baseModelPath, sessionConfiguration ?? SessionConfiguration.DefaultCpu);
		if (File.Exists(modelName)) return modelName;

		String alternativeBaseModelPath = String.Empty;
		String? basePath = Path.GetDirectoryName(Path.GetFullPath(baseModelPath));
		if (!String.IsNullOrEmpty(basePath)) {
			alternativeBaseModelPath = Path.Combine(basePath, "onnx", Path.GetFileName(baseModelPath));
			modelName = GetModelName(alternativeBaseModelPath, sessionConfiguration ?? SessionConfiguration.DefaultCpu);
			if (File.Exists(modelName)) return modelName;
		}

		if (sessionConfiguration != null) {
			if (File.Exists(baseModelPath)) return OptimizeIfNecessary(baseModelPath, sessionConfiguration);
			if (File.Exists(alternativeBaseModelPath)) return OptimizeIfNecessary(alternativeBaseModelPath, sessionConfiguration);
		}

		throw new FileNotFoundException($"Could not find {baseModelPath} in path: {basePath} or onnx subdirectory. Absolute: {Path.GetFullPath(baseModelPath)}");
	}

	private OnnxSession LoadCpu(String modelPath, SessionConfiguration sessionConfiguration, SessionOptions? sharedOptions = null) {
		String modelToLoad = FindPath(modelPath, sessionConfiguration);

		SessionOptions sessionOptions;
		Boolean disposeOptions;
		if (sharedOptions == null) {
			sessionOptions = CreateSessionOptions(sessionConfiguration with { OptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL });
			disposeOptions = true;
		} else {
			sessionOptions = sharedOptions;
			disposeOptions = false;
		}

		_logger.LogDebug("Loading model: {Path}", modelToLoad);
		InferenceSession modelSession = new(modelToLoad, sessionOptions);
		return new OnnxSession(sessionConfiguration, modelSession, sessionOptions, disposeOptions);
	}

	private OnnxSession LoadDml(String modelPath, SessionConfiguration sessionConfiguration, SessionOptions? sharedOptions = null) {
		String modelToLoad = FindPath(modelPath, sessionConfiguration);

		SessionOptions sessionOptions;
		Boolean disposeOptions;
		if (sharedOptions == null) {
			sessionOptions = CreateSessionOptions(sessionConfiguration with { OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL });
			disposeOptions = true;
		} else {
			sessionOptions = sharedOptions;
			disposeOptions = false;
		}

		_logger.LogDebug("Loading model: {Path}", modelToLoad);
		InferenceSession modelSession = new(modelToLoad, sessionOptions);
		return new OnnxSession(sessionConfiguration, modelSession, sessionOptions, disposeOptions);
	}
}

internal class TranslationModelLoader {
	private readonly ILogger _logger;

	public TranslationModelLoader(ILogger logger) {
		_logger = logger;
	}

	public TranslationModel Load(ModelDefinition modelDefinition, SessionConfiguration sessionConfiguration) {
		switch (sessionConfiguration.ExecutionProvider) {
			case ExecutionProvider.CPU:
				return LoadCpu(modelDefinition, sessionConfiguration);
			case ExecutionProvider.DirectML:
				return LoadDml(modelDefinition, sessionConfiguration);
			default:
				throw new ArgumentOutOfRangeException($"Unknown ExecutionProvider: {sessionConfiguration.ExecutionProvider}, must be one of {String.Join(", ", Enum.GetNames(typeof(ExecutionProvider)))}");
		}
	}

	private TranslationModel LoadCpu(ModelDefinition modelDefinition, SessionConfiguration sessionConfiguration) {
		OnnxModelLoader ml = new(_logger);
		OnnxSession encoderSession = ml.Load(Path.Combine(modelDefinition.ModelDirectory, modelDefinition.EncoderModelName), sessionConfiguration);
		OnnxSession decoderSession = ml.Load(Path.Combine(modelDefinition.ModelDirectory, modelDefinition.DecoderModelName), sessionConfiguration, encoderSession.Options);
		OnnxSession decoderWithPastSession = ml.Load(Path.Combine(modelDefinition.ModelDirectory, modelDefinition.DecoderWithPasthModelName), sessionConfiguration, encoderSession.Options);
		String vocabularyFile = Path.Combine(modelDefinition.ModelDirectory, modelDefinition.VocabularyTokenToIdMap);
		String tokenizerFile = Path.Combine(modelDefinition.ModelDirectory, modelDefinition.SourceTokenizer);

		_logger.LogDebug("Loading tokenizer: {Path}", tokenizerFile);
		MarianTokenizer tokenizer = new(tokenizerFile, vocabularyFile);

		return new TranslationModel(modelDefinition, encoderSession, decoderSession, decoderWithPastSession, tokenizer);
	}

	private TranslationModel LoadDml(ModelDefinition modelDefinition, SessionConfiguration sessionConfiguration) {
		throw new NotImplementedException();
		// return new InferenceSession(GetModelName(modelBasename, GraphOptimizationLevel.ORT_DISABLE_ALL), CreateSessionOptions(GraphOptimizationLevel.ORT_ENABLE_ALL, modelBasename, false, null, true), ppwc);
	}
}