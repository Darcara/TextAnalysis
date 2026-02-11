namespace TextAnalysis.Test.LanguageDetection;

using System.Collections.Concurrent;
using System.Diagnostics;
using IsoEnums.Iso639;
using Neco.Common;
using Neco.Common.Extensions;
using TextAnalysis.LanguageDetection;

[TestFixture]
public class AccuracyTests : ATest {
	private readonly Dictionary<String, LanguageTestStats> _statistics = new();

	[SetUp]
	public void BeforeEveryTest() {
		_statistics.Clear();
	}

	[Test]
	public void AccuracyTest() {
		// TestDetector("Lingua-Low", new LinguaLanguageDetector(true));
		TestDetector("Lingua-High", new LinguaLanguageDetector());
		// TestDetector("FastText-Low", new FastTextLanguageDetector(null));
		TestDetector("FastText-176", new FastTextLanguageDetector(TestData.LanguageDetectionModels.FastText176));
		TestDetector("LHigh + FT176", new CombinedDetector(new LinguaLanguageDetector(), new FastTextLanguageDetector(TestData.LanguageDetectionModels.FastText176)));
		TestDetector("FT176 + LHigh", new CombinedDetector(new FastTextLanguageDetector(TestData.LanguageDetectionModels.FastText176), new LinguaLanguageDetector()));
	}

	private IEnumerable<(Language language, String[] entries)> Get(String directory) {
		foreach (String file in Directory.EnumerateFiles($"data/language-testdata/{directory}/", "*.txt")) {
			String fileName = Path.GetFileNameWithoutExtension(file);
			Language language = LanguageHelper.GetLanguageBy2Code(fileName);

			String[] entries = File.ReadAllLines(file);
			yield return (language, entries);
		}
	}

	private IEnumerable<Language> GetTestLanguages(String directory) {
		foreach (String file in Directory.EnumerateFiles($"data/language-testdata/{directory}/", "*.txt")) {
			String fileName = Path.GetFileNameWithoutExtension(file);
			Language language = LanguageHelper.GetLanguageBy2Code(fileName);
			yield return language;
		}
	}

	public void TestDetector(String name, ILanguageDetector detector) {
		_statistics[name] = new LanguageTestStats();
		Dictionary<Language, Int32> testLanguages = GetTestLanguages("single-words").ToDictionary(lang => lang, _ => 0);
		ConcurrentDictionary<Language, Int32> additionalLanguages = new();
		Stopwatch sw = new();
		foreach (String category in new[] { "single-words", "word-pairs", "sentences" }) {
			LanguageTestStats perElementStat = new();
			foreach ((Language language, String[]? entries) in Get(category)) {
				LanguageTestStats perLanguageStat = new();
				sw.Start();
				foreach (String entry in entries) {
					LanguagePrediction languagePrediction = detector.Detect(entry);
					perLanguageStat.Add(languagePrediction, language);
					if (testLanguages.ContainsKey(languagePrediction.Language))
						testLanguages[languagePrediction.Language]++;
					else
						additionalLanguages.AddOrUpdate(languagePrediction.Language, lang => 0, (key, oldValue) => oldValue + 1);
				}

				sw.Stop();
				// Console.WriteLine($"{language, 20}: {perLanguageStat}");
				perElementStat.Add(category, perLanguageStat);
			}

			Console.WriteLine($"{category,12} results for {name,5}: {perElementStat}");
			_statistics[name].Add(category, perElementStat);
		}

		Console.WriteLine($"{"Total",12} results for {name,5}: {_statistics[name]}");
		Console.WriteLine($"Time for {_statistics[name].Predictions} predictions: {sw.Elapsed} or {sw.Elapsed.TotalMilliseconds / _statistics[name].Predictions:N3}ms per prediction");
		Console.WriteLine($"Unsupported Languages: {String.Join(", ", testLanguages.Where(kv => kv.Value == 0).Select(kv => kv.Key))}");
		Console.WriteLine($" Additional Languages: {String.Join(", ", additionalLanguages.Where(kv => kv.Value == 0).Select(kv => kv.Key))}");

		detector.Dispose();
	}
}

internal sealed class CombinedDetector : ILanguageDetector {
	private readonly ILanguageDetector _primary;
	private readonly ILanguageDetector _secondary;
	private readonly Double _boostWithSecondaryConfidenceThreshold;

	public CombinedDetector(ILanguageDetector primary, ILanguageDetector secondary, Double boostWithSecondaryConfidenceThreshold = 0.25) {
		_primary = primary;
		_secondary = secondary;
		_boostWithSecondaryConfidenceThreshold = boostWithSecondaryConfidenceThreshold;
	}

	#region Implementation of ILanguageDetector

	/// <inheritdoc />
	public LanguagePrediction Detect(String text) => Detect(text, 1)[0];

	/// <inheritdoc />
	public LanguagePrediction[] Detect(String text, Int32 count) {
		List<LanguagePrediction> predictions = _primary
			.Detect(text, count)
			.ToList();

		if (predictions[0].Confidence < _boostWithSecondaryConfidenceThreshold) {
			foreach (LanguagePrediction languagePrediction in _secondary.Detect(text, count)) {
				Int32 idx = predictions.FindIndex(p => p.Language == languagePrediction.Language);
				if (idx >= 0)
					predictions[idx] = new LanguagePrediction(predictions[idx].Language, predictions[idx].IsReliable, predictions[idx].Confidence + languagePrediction.Confidence);
				else
					predictions.Add(languagePrediction);
			}
		}

		return predictions.OrderBy(p => p.Confidence).ToArray();
	}

	/// <inheritdoc />
	public LanguagePrediction Detect(ReadOnlySpan<Char> text) => Detect(text, 1)[0];

	/// <inheritdoc />
	public LanguagePrediction[] Detect(ReadOnlySpan<Char> text, Int32 count) => Detect(text.ToString(), count);

	/// <inheritdoc />
	public LanguagePrediction Detect(ReadOnlySpan<Byte> utf8Bytes) => Detect(utf8Bytes, 1)[0];

	/// <inheritdoc />
	public LanguagePrediction[] Detect(ReadOnlySpan<Byte> utf8Bytes, Int32 count) => Detect(MagicNumbers.Utf8NoBom.GetString(utf8Bytes), count);

	#endregion

	#region IDisposable

	/// <inheritdoc />
	public void Dispose() {
		_primary.Dispose();
		_secondary.Dispose();
	}

	#endregion
}