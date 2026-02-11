namespace TextAnalysis.LanguageDetection;

using System.Collections.Frozen;
using IsoEnums.Iso639;
using Neco.Common;
using Panlingo.LanguageIdentification.FastText;

public sealed class FastTextLanguageDetector : ILanguageDetector {
	private readonly FastTextDetector _detector;
	private static readonly FrozenDictionary<String, Language> _lookup2Code = LanguageHelper.CreateFast2CodeLookup();
	private static readonly FrozenDictionary<String, Language>.AlternateLookup<ReadOnlySpan<Char>> _lookup2CodeAlt = _lookup2Code.GetAlternateLookup<ReadOnlySpan<Char>>();
	private static readonly FrozenDictionary<String, Language> _lookup3Code = LanguageHelper.CreateFast3CodeLookup();
	private static readonly FrozenDictionary<String, Language>.AlternateLookup<ReadOnlySpan<Char>> _lookup3CodeAlt = _lookup3Code.GetAlternateLookup<ReadOnlySpan<Char>>();

	public FastTextLanguageDetector(String? modelFile) {
		_detector = new FastTextDetector();

		if (modelFile is null) {
			_detector.LoadDefaultModel();
			return;
		}

		FileInfo fileInfo = new(modelFile);
		if (!fileInfo.Exists)
			throw new FileNotFoundException($"File {fileInfo.FullName} not found", modelFile);

		_detector.LoadModel(fileInfo.FullName);
	}

	internal IEnumerable<String> GetFastTextLabels() => _detector.GetLabels().Select(label => label.Label);

	internal Language MapFastTextLabel(String label) {
		// __label__xx or __label__xxx for two / three codes (for 176)
		// __label__xxx_Latn three code (for 218e)
		ReadOnlySpan<Char> code = label.AsSpan(9);
		if(code.Length > 3) code = code.Slice(0, 3);

		// EML split into Emilian(3.3 Mio ethnic, 1.3 Mio spakers) and Romagnol(~1.1 Mio ethnic, 430k speakers)
		if (code is "eml") return Language.Emilian;

		// Proper code is bih, bh is deprecated
		if (code is "bh") return Language.Bihari_languages;

		if (code.Length == 2)
			return _lookup2CodeAlt.TryGetValue(code, out Language language) ? language : Language.Undetermined;

		if (code.Length == 3)
			return _lookup3CodeAlt.TryGetValue(code, out Language language) ? language : Language.Undetermined;

		return Language.Undetermined;
	}

	#region Implementation of ILanguageDetector

	/// <inheritdoc />
	public LanguagePrediction Detect(String text) => Detect(text, 1)[0];

	/// <inheritdoc />
	public LanguagePrediction[] Detect(String text, Int32 count) {
		LanguagePrediction[] predictions = _detector
			.Predict(text, count, -1)
			.Select(pred => new LanguagePrediction(MapFastTextLabel(pred.Label), pred.Probability > 0.25, pred.Probability))
			.ToArray();
		
		if(predictions.Length > 0) return predictions;
		return [new LanguagePrediction(Language.Undetermined, false, 0)];
	}

	/// <inheritdoc />
	public LanguagePrediction Detect(ReadOnlySpan<Char> text) => Detect(text.ToString(), 1)[0];

	/// <inheritdoc />
	public LanguagePrediction[] Detect(ReadOnlySpan<Char> text, Int32 count) => Detect(text.ToString(), count);

	/// <inheritdoc />
	public LanguagePrediction Detect(ReadOnlySpan<Byte> utf8Bytes) => Detect(utf8Bytes, 1)[0];

	/// <inheritdoc />
	public LanguagePrediction[] Detect(ReadOnlySpan<Byte> utf8Bytes, Int32 count) {
		String text = MagicNumbers.Utf8NoBom.GetString(utf8Bytes);
		return Detect(text, count);
	}

	#endregion

	#region IDisposable

	/// <inheritdoc />
	public void Dispose() {
		_detector.Dispose();
	}

	#endregion
}