namespace TextAnalysis.LanguageDetection;

public sealed class LinguaLanguageDetector : ILanguageDetector {
	private readonly CustomLinguaDetector _detector;
	
	/// <summary>
	/// 
	/// </summary>
	/// <param name="useLowAccuracy">Enables low accuracy mode. Low accuracy is roughly 3 times faster, but has 3% lower accuracy for sentences (96% -&gt; 93%) and 11% for word pairs (89% -&gt; 78%)</param>
	public LinguaLanguageDetector(Boolean useLowAccuracy = false) {
		_detector = new CustomLinguaDetector(useLowAccuracy);
	}

	public LanguagePrediction Detect(String text) => _detector.PredictLanguages(text, 1)[0];

	public LanguagePrediction[] Detect(String text, Int32 count) => _detector.PredictLanguages(text, count);

	/// <inheritdoc />
	public LanguagePrediction Detect(ReadOnlySpan<Char> text) => _detector.PredictLanguages(text, 1)[0];

	/// <inheritdoc />
	public LanguagePrediction[] Detect(ReadOnlySpan<Char> text, Int32 count) => _detector.PredictLanguages(text, count);

	/// <inheritdoc />
	public LanguagePrediction Detect(ReadOnlySpan<Byte> utf8Bytes) => _detector.PredictLanguages(utf8Bytes, 1)[0];

	/// <inheritdoc />
	public LanguagePrediction[] Detect(ReadOnlySpan<Byte> utf8Bytes, Int32 count) => _detector.PredictLanguages(utf8Bytes, count);


	#region IDisposable

	/// <inheritdoc />
	public void Dispose() {
		_detector.Dispose();
	}

	#endregion
}