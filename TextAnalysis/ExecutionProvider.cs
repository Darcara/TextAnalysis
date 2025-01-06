namespace TextAnalysis;

using System.Diagnostics.CodeAnalysis;

// Only one runtime can be referenced at a time?? https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages
// this might be fixed by https://github.com/magiccodingman/MagicOnnxRuntimeGenAi
// All include the CPUExecutor so Microsoft.ML.OnnxRuntime itself is not required
// Register execution providers: https://onnxruntime.ai/docs/execution-providers/
[SuppressMessage("ReSharper", "InconsistentNaming")]
public enum ExecutionProvider {
	/// <summary>
	/// Uses the CPU only.
	/// </summary>
	/// <remarks>A single CPU model can be used concurrently from multiple threads.</remarks>
	CPU = 0,
	
	/// <summary>
	/// Direct Machine Learning (DirectML) is a low-level API for machine learning (ML) on the GPU.
	/// DirectML is supported by all DirectX 12-compatible hardware.
	/// </summary>
	/// <seealso href="https://learn.microsoft.com/en-us/windows/ai/directml/dml"/>
	/// <seealso href="https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html#configuration-options"/>
	/// <remarks>DirectML models can not be used concurrently.</remarks>
	DirectML = 1,
}