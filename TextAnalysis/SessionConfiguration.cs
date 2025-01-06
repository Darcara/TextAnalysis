namespace TextAnalysis;

using Microsoft.ML.OnnxRuntime;

public sealed record BatchingConfiguration {
	public static readonly BatchingConfiguration NoBatching = new();

	public Int64 BatchSize { get; init; } = 1;

	public String? BatchingDimensionName { get; init; }
}

public enum DimensionOverrideType {
	ByName = 0,
	ByDenotation,
}

public sealed record FreeDimensionOverride {
	public DimensionOverrideType Type { get; init; }
	public String Key { get; init; }
	public Int64 Value { get; init; }

	public FreeDimensionOverride(DimensionOverrideType type, String key, Int64 value) {
		Type = type;
		Key = key;
		Value = value;
	}
}

public sealed record SessionConfiguration {
	public static readonly SessionConfiguration DefaultCpu = new();

	public ExecutionProvider ExecutionProvider { get; init; } = ExecutionProvider.CPU;

	public GraphOptimizationLevel OptimizationLevel { get; init; } = GraphOptimizationLevel.ORT_DISABLE_ALL;

	public BatchingConfiguration Batching { get; init; } = BatchingConfiguration.NoBatching;

	public List<FreeDimensionOverride>? FreeDimensionOverrides { get; init; }

	/// <summary>
	/// Set which GPU-Device to use, default: 0
	/// </summary>
	/// <remarks>Will be ignored for <see cref="TextAnalysis.ExecutionProvider.CPU"/> based <see cref="ExecutionProvider"/>s</remarks>
	public Int32 GpuDeviceId { get; init; } = 0;

	/// <summary>
	/// The number of threads used to parallelize the execution within nodes/operators
	/// </summary>
	public Int32 IntraOpNumThreads { get; init; } = 1;

	/// <summary>
	/// The Number of threads used to parallelize the execution of the graph (across nodes)
	/// </summary>
	public Int32 InterOpNumThreads { get; init; } = 1;

	/// <summary>
	/// Logging will produce (a lot) console output, especially during model loading, default: false
	/// <para>Verbose output can be very helpful to determine model optimization results and node placement (CPU/GPU)</para>
	/// </summary>
	public Boolean EnableVerboseOrtLogging { get; init; } = false;
	// TODO LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL,
	// TODO LogVerbosityLevel = 0,
	// LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE,
	// LogVerbosityLevel = 100,

	// if (withProfiling) {
	// 	opt.EnableProfiling = true;
	// 	opt.ProfileOutputPathPrefix = profilingPathPrefix;
	// }

	/// <summary>
	/// Enable or disable gelu approximation in graph optimization. <br/>
	/// GeluApproximation has side effects which may change the inference results. It is disabled by default due to this.
	/// </summary>
	/// <seealso href="https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h"/>
	public Boolean EnableGeluApproximation { get; init; } = false;

	/// <summary>
	/// GEMM == General Matrix multiplication <br/>
	/// Gemm fastmath mode provides fp32 gemm acceleration with bfloat16 based matmul.<br/>
	/// Default: disabled
	/// </summary>
	/// <seealso href="https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h"/>
	public Boolean EnableGemmFastMath { get; init; } = false;

	/// <summary>
	/// This setting controls whether to disable AheadOfTime function inlining. Default: false, AOT function inlining is enabled<br/>
	/// AOT function inlining examines the graph and attempts to inline as many locally defined functions in the model as possible with the help of enabled execution providers.
	/// This can reduce the number of function calls and improve performance because it is done before
	/// Level1 optimizers and constant folding. However, under some circumstances, when the EPs are not available,
	/// one can disable the AOT inlining, produce an optimized model and postpone AOT until run time.
	/// </summary>
	/// <seealso href="https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h"/>
	public Boolean DisableAheadOfTimeFunctionInlining { get; init; } = false;

	/// <summary>
	/// Enable or disable using device allocator for allocating initialized tensor memory. Default: disabled. <br/>
	/// Using device allocators means the memory allocation is made using malloc/new.
	/// </summary>
	/// <seealso href="https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h"/>
	public Boolean UseDeviceAllocatorForInitializers { get; init; } = false;

	/// <summary>
	/// This Option allows setting affinities for <see cref="IntraOpNumThreads">intra op threads</see>. <br/>
	/// The default is null, meaning no affinities specified and all cores may be used.
	/// <code>
	/// Affinity string follows format:
	/// logical_processor_id,logical_processor_id;logical_processor_id,logical_processor_id
	/// Semicolon isolates configurations among threads, while comma split processors where ith thread expected to attach to.
	/// e.g.1,2,3;4,5
	/// specifies affinities for two threads, with the 1st thread attach to the 1st, 2nd, and 3rd processor, and 2nd thread to the 4th and 5th.
	/// To ease the configuration, an "interval" is also allowed:
	/// e.g. 1-8;8-16;17-24
	/// orders that the 1st thread runs on first eight processors, 2nd thread runs on next eight processors, and so forth.
	/// </code>
	/// </summary>
	/// <remarks>
	/// <p>1. Once set, the number of thread affinities must equal to intra_op_num_threads - 1, since ort does not set affinity on the main thread which
	///    is started and managed by the calling app;</p> 
	/// <p>2. For windows, ort will infer the group id from a logical processor id, for example, assuming there are two groups with each has 64 logical processors,
	///    an id of 64 will be inferred as the last processor of the 1st group, while 65 will be interpreted as the 1st processor of the second group.
	///    Hence 64-65 is an invalid configuration, because a windows thread cannot be attached to processors across group boundary.</p>
	/// </remarks>
	/// <seealso href="https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h"/>
	public String? IntraOpThreadAffinities { get; init; } = null;
}