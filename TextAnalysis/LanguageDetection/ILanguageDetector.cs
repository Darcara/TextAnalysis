namespace TextAnalysis.LanguageDetection;

public interface ILanguageDetector : IDisposable {
	/// <summary>
	/// Detect the most likely language
	/// </summary>
	public LanguagePrediction Detect(String text);

	/// <summary>
	/// Detect the top-n most likely languages
	/// </summary>
	public LanguagePrediction[] Detect(String text, Int32 count);

	/// <inheritdoc cref="Detect(string)"/>
	public LanguagePrediction Detect(ReadOnlySpan<Char> text);

	/// <inheritdoc cref="Detect(string,int)"/>
	public LanguagePrediction[] Detect(ReadOnlySpan<Char> text, Int32 count);
	
	/// <summary>
	/// Detect the most likely language from the given UTF-8 byte-span
	/// </summary>
	public LanguagePrediction Detect(ReadOnlySpan<Byte> utf8Bytes);

	/// <summary>
	/// Detect the top-n most likely languages from the given UTF-8 byte-span
	/// </summary>
	public LanguagePrediction[] Detect(ReadOnlySpan<Byte> utf8Bytes, Int32 count);
}