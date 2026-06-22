namespace TextAnalysis;

using System.IO.Hashing;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Neco.Common;
using Neco.Common.Extensions;
using SentencePieceTokenizer;
using TextAnalysis.Translation;

public sealed class OnnxSession : IDisposable {
	private readonly Boolean _disposeOptions;
	private readonly Boolean _disposeOrtValueCache;
	internal readonly OrtValueCache? OrtValueCache;
	private readonly List<(String, OrtValue)> _additionalInputs;
	internal readonly SessionOptions Options;
	public readonly SessionConfiguration Configuration;
	public readonly InferenceSession Session;
	public readonly RunOptions RunOptions;
	public readonly String Name;

	public OnnxSession(String name, SessionConfiguration configuration, InferenceSession session, SessionOptions options, Boolean disposeOptions, OrtValueCache? ortValueCache, List<(String, OrtValue)>? additionalInputs, Boolean disposeOrtValueCache) {
		Name = name;
		Configuration = configuration;
		Session = session;
		Options = options;
		_disposeOptions = disposeOptions;
		OrtValueCache = ortValueCache;
		_additionalInputs = additionalInputs ?? [];
		_disposeOrtValueCache = disposeOrtValueCache;
		RunOptions = new RunOptions();
	}

	public Dictionary<string, OrtValue> CreateStaticInputs(Int32 capacity = 0) {
		Dictionary<String, OrtValue> inputs = new(_additionalInputs.Count + capacity, StringComparer.Ordinal);
		for (int index = 0; index < _additionalInputs.Count; index++) {
			(String item1, OrtValue ortValue) = _additionalInputs[index];
			inputs.Add(item1, ortValue);
		}

		return inputs;
	}
	
	public OrtValue[] CreateStaticInputArray(Int32 capacity) {
		OrtValue[] inputs = new OrtValue[_additionalInputs.Count + capacity];
		for (int index = 0; index < _additionalInputs.Count; index++) {
			(String _, OrtValue ortValue) = _additionalInputs[index];
			inputs[capacity + index]= ortValue;
		}

		return inputs;
	}

	#region IDisposable

	/// <inheritdoc />
	public void Dispose() {
		RunOptions.Dispose();
		Session.Dispose();
		if (_disposeOptions)
			Options.Dispose();
		if (_disposeOrtValueCache)
			OrtValueCache.Dispose();
	}

	#endregion
}

internal class OnnxModelLoader {
	private readonly ILogger _logger;

	public OnnxModelLoader(ILogger? logger = null) {
		_logger = logger ?? NullLogger.Instance;
	}

	public OnnxSession Load(String modelPath, SessionConfiguration sessionConfiguration, SessionOptions? sharedOptions = null, OrtValueCache? ortValueCache = null) {
		switch (sessionConfiguration.ExecutionProvider) {
			case ExecutionProvider.CPU:
				return LoadCpu(modelPath, sessionConfiguration, sharedOptions, ortValueCache);
			case ExecutionProvider.DirectML:
				return LoadDml(modelPath, sessionConfiguration, sharedOptions, ortValueCache);
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
		};
		if (!String.IsNullOrEmpty(optimizedOutput))
			opt.OptimizedModelFilePath = optimizedOutput;
		opt.RegisterOrtExtensions();
		opt.SetEpSelectionPolicy(ExecutionProviderDevicePolicy.MAX_PERFORMANCE);

		// TODO LogLevel
		if (sessionConfiguration.EnableVerboseOrtLogging) {
			opt.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
			opt.LogVerbosityLevel = 100;
		}
		// TODO profiling
		// opt.EnableProfiling = true;

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
					opt.SetEpSelectionPolicy(ExecutionProviderDevicePolicy.PREFER_GPU);

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
			// case ExecutionProvider.Cuda:
			// 	opt.SetEpSelectionPolicy(ExecutionProviderDevicePolicy.PREFER_GPU);
			//
			// 	// https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#c
			// 	opt.EnableMemoryPattern = false;
			// 	opt.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
			// 	opt.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
			// 	// opt.AppendExecutionProvider_DML(sessionConfiguration.GpuDeviceId);
			// 	opt.AppendExecutionProvider_CUDA(sessionConfiguration.GpuDeviceId);
			// 	// CPU fallback has to be defined AFTER DML endpoint
			// 	// CPU endpoint is always implicitly available, but may generate ORT warnings during model load, such as
			// 	// "VerifyEachNodeIsAssignedToAnEp": Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf. 
			// 	opt.AppendExecutionProvider_CPU();
			//
			// 	var asdf = new OrtCUDAProviderOptions() { };
			// 	SessionOptions options = SessionOptions.MakeSessionOptionWithCudaProvider(asdf);
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
		keyBuilder.AppendLine($"x64: {Environment.Is64BitProcess}");
		keyBuilder.AppendLine($"MachineName: {Environment.MachineName}");
		keyBuilder.AppendLine($"Framework: {RuntimeInformation.FrameworkDescription}");
		keyBuilder.AppendLine($"Runtime: {RuntimeInformation.RuntimeIdentifier}");
		keyBuilder.AppendLine($"Onnx: {OrtEnv.Instance().GetVersionString()}");


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

	private OnnxSession LoadCpu(String modelPath, SessionConfiguration sessionConfiguration, SessionOptions? sharedOptions = null, OrtValueCache? sharedValueCache = null) {
		String modelToLoad = FindPath(modelPath, sessionConfiguration);
		String externalInitializerFile = Path.ChangeExtension(modelPath, ".json");
		Boolean hasExternalInitializer = File.Exists(externalInitializerFile);
		OrtValueCache? cache = null;
		List<(String, OrtValue)>? additionalInputs = null;
		Boolean disposeCache = false;

		if (hasExternalInitializer) {
			_logger.LogDebug("Loading initializer definitions from from {ExternalInitializerFile}", externalInitializerFile);
			Dictionary<String, InitializerDefinition> intializerNames = JsonSerializer.Deserialize<Dictionary<String, InitializerDefinition>>(File.ReadAllText(externalInitializerFile)) ?? throw new InvalidOperationException($"Unable to read {externalInitializerFile}");
			if (sharedValueCache == null) {
				cache = new();
				string initializerDirectory = Path.Combine(Path.GetDirectoryName(modelPath) ?? ".", "initializers");
				_logger.LogDebug("Loading initializers from {InitializerDirectory}", initializerDirectory);
				cache.LoadInitializers(initializerDirectory);
				disposeCache = true;
			} else {
				cache = sharedValueCache;
				disposeCache = false;
			}

			additionalInputs = new(intializerNames.Count);

			foreach ((String name, InitializerDefinition def) in intializerNames) {
				// _logger.LogDebug($"Initializer added: {name} with {def.Type}[{String.Join(",", def.Shape ?? [])}]");
				if (!Enum.TryParse(def.Type, true, out TensorElementType elementType))
					throw new InvalidOperationException($"Unable to load initializer {def.File}, cannot parse type:{def.Type}");
				additionalInputs.Add((name, cache.Register(def.File, elementType, def.Shape)));
			}
		}

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
		return new OnnxSession(Path.GetFileName(modelToLoad), sessionConfiguration, modelSession, sessionOptions, disposeOptions, cache, additionalInputs, disposeCache);
	}

	private OnnxSession LoadDml(String modelPath, SessionConfiguration sessionConfiguration, SessionOptions? sharedOptions = null, OrtValueCache? sharedValueCache = null) {
		return LoadCpu(modelPath, sessionConfiguration with { OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL }, sharedOptions, sharedValueCache);
		// String modelToLoad = FindPath(modelPath, sessionConfiguration);
		//
		// SessionOptions sessionOptions;
		// Boolean disposeOptions;
		// if (sharedOptions == null) {
		// 	sessionOptions = CreateSessionOptions(sessionConfiguration with { OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL });
		// 	disposeOptions = true;
		// } else {
		// 	sessionOptions = sharedOptions;
		// 	disposeOptions = false;
		// }
		//
		// _logger.LogDebug("Loading model: {Path}", modelToLoad);
		// InferenceSession modelSession = new(modelToLoad, sessionOptions);
		// return new OnnxSession(Path.GetFileName(modelToLoad), sessionConfiguration, modelSession, sessionOptions, disposeOptions, null, null, false);
	}
}

internal class TranslationModelLoader {
	private readonly ILogger _logger;
	private readonly OnnxModelLoader ModelLoader;

	public TranslationModelLoader(OnnxModelLoader? modelLoader = null, ILogger? logger = null) {
		_logger = logger ?? NullLogger.Instance;
		ModelLoader = modelLoader ?? new OnnxModelLoader(_logger);
	}

	public ITranslator Load(ModelDefinition modelDefinition, SessionConfiguration sessionConfiguration) {
		switch (sessionConfiguration.ExecutionProvider) {
			case ExecutionProvider.CPU:
				return LoadCpu(modelDefinition, sessionConfiguration);
			case ExecutionProvider.DirectML:
				return LoadDml(modelDefinition, sessionConfiguration);
			default:
				throw new ArgumentOutOfRangeException($"Unknown ExecutionProvider: {sessionConfiguration.ExecutionProvider}, must be one of {String.Join(", ", Enum.GetNames(typeof(ExecutionProvider)))}");
		}
	}

	private ITranslator LoadCpu(ModelDefinition modelDefinition, SessionConfiguration sessionConfiguration) {
		String vocabularyFile = Path.Combine(modelDefinition.ModelDirectory, modelDefinition.VocabularyTokenToIdMap);
		String tokenizerFile = Path.Combine(modelDefinition.ModelDirectory, modelDefinition.SourceTokenizer);

		_logger.LogDebug("Loading tokenizer: {Path}", tokenizerFile);
		MarianTokenizer tokenizer = new(tokenizerFile, vocabularyFile);

		switch (modelDefinition.Type) {
			case TranslationModelType.Separate3ModelsWithPast: {
				OnnxSession encoderSession = ModelLoader.Load(Path.Combine(modelDefinition.ModelDirectory, modelDefinition.EncoderModelName), sessionConfiguration);
				OnnxSession decoderSession = ModelLoader.Load(Path.Combine(modelDefinition.ModelDirectory, modelDefinition.DecoderModelName), sessionConfiguration, encoderSession.Options, encoderSession.OrtValueCache);
				OnnxSession decoderWithPastSession = ModelLoader.Load(Path.Combine(modelDefinition.ModelDirectory, modelDefinition.DecoderWithPastModelName), sessionConfiguration, encoderSession.Options, encoderSession.OrtValueCache);

				return new TranslationModel3WithPast(modelDefinition, encoderSession, decoderSession, decoderWithPastSession, tokenizer);
			}
			case TranslationModelType.Separate2ModelsWithPast: {
				OnnxSession encoderSession = ModelLoader.Load(Path.Combine(modelDefinition.ModelDirectory, modelDefinition.EncoderModelName), sessionConfiguration);
				OnnxSession decoderWithPastSession = ModelLoader.Load(Path.Combine(modelDefinition.ModelDirectory, modelDefinition.DecoderWithPastModelName), sessionConfiguration, encoderSession.Options, encoderSession.OrtValueCache);

				return new TranslationModel2WithPast(modelDefinition, tokenizer, encoderSession, decoderWithPastSession);
			}
			case TranslationModelType.Undefined:
			default:
				throw new ArgumentOutOfRangeException(nameof(modelDefinition.Type), modelDefinition.Type, $"Unknown {nameof(TranslationModelType)}: {modelDefinition.Type})");
		}
	}

	private ITranslator LoadDml(ModelDefinition modelDefinition, SessionConfiguration sessionConfiguration) {
		return LoadCpu(modelDefinition, sessionConfiguration);
	}
}