namespace TextAnalysis.LanguageDetection;

using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using System.Text;
using Panlingo.LanguageIdentification.Lingua;

[ExcludeFromCodeCoverage]
internal readonly struct LinguaPredictionListResult {
	public readonly IntPtr Predictions;
	public readonly int PredictionsCount;
}

[ExcludeFromCodeCoverage]
[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
internal readonly struct LinguaPredictionResult {
	[MarshalAs(UnmanagedType.U1)] public readonly LinguaLanguage Language;

	[MarshalAs(UnmanagedType.R8)] public readonly double Confidence;
}

[ExcludeFromCodeCoverage]
internal static partial class LinguaDetectorWrapper {
	[DllImport("lingua", EntryPoint = "lingua_prediction_result_destroy", CallingConvention = CallingConvention.Cdecl)]
	public static extern void LinguaPredictionResultDestroy(IntPtr result);

	[DllImport("lingua", EntryPoint = "lingua_prediction_range_result_destroy", CallingConvention = CallingConvention.Cdecl)]
	public static extern void LinguaPredictionRangeResultDestroy(IntPtr result);

	// [DllImport("lingua", EntryPoint = "lingua_detect_single", CallingConvention = CallingConvention.Cdecl)]
	[LibraryImport("lingua", EntryPoint = "lingua_detect_single")]
	public static partial LinguaStatus LinguaDetectSingle(IntPtr detector, byte[] text, UIntPtr textLength, out LinguaPredictionListResult result);

	
	[LibraryImport("lingua", EntryPoint = "lingua_detect_single")]
	public static partial LinguaStatus LinguaDetectSingle(IntPtr detector, ReadOnlySpan<Byte> text, UIntPtr textLength, out LinguaPredictionListResult result);

	[LibraryImport("lingua", EntryPoint = "lingua_detect_single")]
	public static partial LinguaStatus LinguaDetectSingle(IntPtr detector, IntPtr text, UIntPtr textLength, out LinguaPredictionListResult result);

	// [DllImport("lingua", EntryPoint = "lingua_detect_mixed", CallingConvention = CallingConvention.Cdecl)]
	// public static extern LinguaStatus LinguaDetectMixed(IntPtr detector, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, out LinguaPredictionRangeListResult result);

	[DllImport("lingua", EntryPoint = "lingua_language_code", CallingConvention = CallingConvention.Cdecl)]
	public static extern int LinguaLangCode(LinguaLanguage lang, LinguaLanguageCode code, [MarshalAs(UnmanagedType.LPUTF8Str)] StringBuilder buffer, UIntPtr bufferSize);

	public static unsafe IntPtr ConvertStringToNativeUtf8ZeroTerminated(ReadOnlySpan<Char> managedString, out Int32 numBytes) {
		// from Utf8StringMarshaller.ConvertToUnmanaged but with processHeap instead of COM-Heap
		Int32 exactByteCount = Encoding.UTF8.GetByteCount(managedString) + 1; // + 1 for null terminator
		Byte* mem = (Byte*)Marshal.AllocHGlobal(exactByteCount);
		Span<Byte> buffer = new(mem, exactByteCount);

		Int32 byteCount = Encoding.UTF8.GetBytes(managedString, buffer);
		buffer[byteCount] = 0;
		numBytes = byteCount;
		return new IntPtr(mem);
	}
}