namespace TextAnalysis.LanguageDetection;

using IsoEnums.Iso639;

public readonly record struct LanguagePrediction(Language Language, Boolean IsReliable, Double Confidence) {
	/// <summary>
	/// The predicted language
	/// </summary>
	public readonly Language Language = Language;

	/// <summary>
	/// Detector specific reliabilty, usually if the <see cref="Confidence"/> is above a certain threshold
	/// </summary>
	public readonly Boolean IsReliable = IsReliable;

	/// <summary>
	/// Confidence of the prediction in the interval 0 - 1
	/// </summary>
	public readonly Double Confidence = Confidence;
}