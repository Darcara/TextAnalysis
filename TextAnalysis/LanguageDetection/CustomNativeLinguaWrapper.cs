namespace TextAnalysis.LanguageDetection;

using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using System.Text;
using Panlingo.LanguageIdentification.Lingua;

[ExcludeFromCodeCoverage]
internal readonly struct LinguaPredictionListResult
{
	public readonly IntPtr Predictions;
	public readonly int PredictionsCount;
}

[ExcludeFromCodeCoverage]
[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
internal readonly struct LinguaPredictionResult
{
	[MarshalAs(UnmanagedType.U1)]
	public readonly LinguaLanguage Language;

	[MarshalAs(UnmanagedType.R8)]
	public readonly double Confidence;
}

internal static partial class LinguaDetectorWrapper {
	[DllImport("lingua", EntryPoint = "lingua_language_detector_builder_create", CallingConvention = CallingConvention.Cdecl)]
	public static extern IntPtr LinguaLanguageDetectorBuilderCreate(LinguaLanguage[] languages, UIntPtr languageCount);

	[DllImport("lingua", EntryPoint = "lingua_language_detector_builder_with_low_accuracy_mode", CallingConvention = CallingConvention.Cdecl)]
	public static extern IntPtr LinguaLanguageDetectorBuilderWithLowAccuracyMode(IntPtr builder);

	[DllImport("lingua", EntryPoint = "lingua_language_detector_builder_with_preloaded_language_models", CallingConvention = CallingConvention.Cdecl)]
	public static extern IntPtr LinguaLanguageDetectorBuilderWithPreloadedLanguageModels(IntPtr builder);

	[DllImport("lingua", EntryPoint = "lingua_language_detector_builder_with_minimum_relative_distance", CallingConvention = CallingConvention.Cdecl)]
	public static extern IntPtr LinguaLanguageDetectorBuilderWithMinimumRelativeDistance(IntPtr builder, double distance);

	[DllImport("lingua", EntryPoint = "lingua_language_detector_create", CallingConvention = CallingConvention.Cdecl)]
	public static extern IntPtr LinguaLanguageDetectorCreate(IntPtr builder);

	[DllImport("lingua", EntryPoint = "lingua_language_detector_builder_destroy", CallingConvention = CallingConvention.Cdecl)]
	public static extern void LinguaLanguageDetectorBuilderDestroy(IntPtr builder);

	[DllImport("lingua", EntryPoint = "lingua_language_detector_destroy", CallingConvention = CallingConvention.Cdecl)]
	public static extern void LinguaLanguageDetectorDestroy(IntPtr detector);

	[DllImport("lingua", EntryPoint = "lingua_prediction_result_destroy", CallingConvention = CallingConvention.Cdecl)]
	public static extern void LinguaPredictionResultDestroy(IntPtr result);

	[DllImport("lingua", EntryPoint = "lingua_prediction_range_result_destroy", CallingConvention = CallingConvention.Cdecl)]
	public static extern void LinguaPredictionRangeResultDestroy(IntPtr result);

	// [DllImport("lingua", EntryPoint = "lingua_detect_single", CallingConvention = CallingConvention.Cdecl)]
	[LibraryImport("lingua", EntryPoint = "lingua_detect_single", StringMarshalling = StringMarshalling.Utf8)]
	public static partial LinguaStatus LinguaDetectSingle(IntPtr detector, string text, out LinguaPredictionListResult result);

	[LibraryImport("lingua", EntryPoint = "lingua_detect_single")]
	public static partial LinguaStatus LinguaDetectSingle2(IntPtr detector, ReadOnlySpan<Byte> utf8, out LinguaPredictionListResult result);

	// [DllImport("lingua", EntryPoint = "lingua_detect_mixed", CallingConvention = CallingConvention.Cdecl)]
	// public static extern LinguaStatus LinguaDetectMixed(IntPtr detector, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, out LinguaPredictionRangeListResult result);

	[DllImport("lingua", EntryPoint = "lingua_language_code", CallingConvention = CallingConvention.Cdecl)]
	public static extern int LinguaLangCode(LinguaLanguage lang, LinguaLanguageCode code, [MarshalAs(UnmanagedType.LPUTF8Str)] StringBuilder buffer, UIntPtr bufferSize);
}