namespace TextAnalysis.Test.LanguageDetection;

using System.Collections.Concurrent;
using System.Collections.Frozen;
using System.Diagnostics;
using GeoInfo.Iso639;
using Neco.Common.Helper;
using Panlingo.LanguageCode;
using Panlingo.LanguageIdentification.CLD2;
using Panlingo.LanguageIdentification.CLD3;
using Panlingo.LanguageIdentification.FastText;
using Panlingo.LanguageIdentification.Lingua;
using Panlingo.LanguageIdentification.Whatlang;

public record struct LanguagePrediction {
	public Language Language;
	public bool IsReliable;
	public double Confidence;

	public LanguagePrediction(Language language, Boolean isReliable, Double confidence) {
		Language = language;
		IsReliable = isReliable;
		Confidence = confidence;
	}
}

internal class LanguageTestStats {
	public Int64 Predictions;
	public Int64 Positive;
	public Int64 Negative;
	public Int64 Unknown;

	public void Add(LanguagePrediction languagePrediction, Language correctLanguage) {
		Predictions++;
		if (languagePrediction.Language == correctLanguage)
			++Positive;
		else if (languagePrediction.Language == SpecialLanguageCodes.Undetermined)
			++Unknown;
		else
			++Negative;
	}

	public void Add(string category, LanguageTestStats languageTestStats) {
		Predictions += languageTestStats.Predictions;
		Positive += languageTestStats.Positive;
		Negative += languageTestStats.Negative;
		Unknown += languageTestStats.Unknown;
	}

	#region Overrides of Object

	/// <inheritdoc />
	public override String ToString() => $"Accuracy = {Positive,4}/{Predictions}={(Positive / (Double)Predictions),7:P2} -- Negative = {(Negative / (Double)Predictions),7:P2} -- Unknown = {(Unknown / (Double)Predictions),7:P2} ==> Reliability = {((Positive + Unknown) / (Double)Predictions),7:P2}";

	#endregion
}

public class Evaluation : ATest {
	private readonly Dictionary<String, LanguageTestStats> _statistics = new();

	private IEnumerable<(Language language, String[] entries)> Get(String directory) {
		foreach (string file in Directory.EnumerateFiles($"data/language-testdata/{directory}/", "*.txt")) {
			string fileName = Path.GetFileNameWithoutExtension(file);
			Language language = LanguageHelper.GetLanguageBy2Code(fileName);

			var entries = File.ReadAllLines(file);
			yield return (language, entries);
		}
	}
	private IEnumerable<Language> GetTestLanguages(String directory) {
		foreach (string file in Directory.EnumerateFiles($"data/language-testdata/{directory}/", "*.txt")) {
			string fileName = Path.GetFileNameWithoutExtension(file);
			Language language = LanguageHelper.GetLanguageBy2Code(fileName);
			yield return language;
		}
	}

	[Test]
	public void SingleWord() {
		FrozenDictionary<string, Language> fast2CodeLookup = LanguageHelper.CreateFast2CodeLookup();
		FrozenDictionary<string, Language> fast3CodeLookup = LanguageHelper.CreateFast3CodeLookup();
		TestDetector("CLD2", () => new CLD2Detector(), (cld2, entry) => {
			IEnumerable<CLD2Prediction> predictions = cld2.PredictLanguage(entry);
			// foreach (CLD2Prediction prediction in predictions) Console.WriteLine($"Language: {fast2CodeLookup.GetValueOrDefault(prediction.Language, Language.Undetermined)}, Probability: {prediction.Probability}, IsReliable: {prediction.IsReliable}, Proportion: {prediction.Proportion}");
			CLD2Prediction first = predictions.First();
			return new LanguagePrediction(fast2CodeLookup.GetValueOrDefault(first.Language, Language.Undetermined), first.IsReliable, first.Proportion);
		});
		TestDetector("CLD3", () => new CLD3Detector(0, 512), (cld3, entry) => {
			// foreach (CLD3Prediction prediction in cld3.PredictLanguages(entry, 3)) Console.WriteLine($"Language: {fast2CodeLookup.GetValueOrDefault(prediction.Language, Language.Undetermined)}, Probability: {prediction.Probability}, IsReliable: {prediction.IsReliable}, Proportion: {prediction.Proportion}");
			CLD3Prediction first = cld3.PredictLanguage(entry);
			return new LanguagePrediction(fast2CodeLookup.GetValueOrDefault(first.Language, Language.Undetermined), first.IsReliable, first.Probability);
		});

		Func<String, Language> fastTextLabelToLanguage = lbl => {
			String code = lbl.Substring(9);
			if (code.IndexOf('_') > 0)
				code = code.Substring(0, code.IndexOf('_'));
			Language l;
			if (code == "eml") l = Language.Emilian;
			else if (code == "nah") l = fast3CodeLookup.GetValueOrDefault("nhe", Language.Undetermined);
			else if (code == "bh") l = Language.Bhojpuri;
			else
				l = code.Length == 2 ? fast2CodeLookup.GetValueOrDefault(code, Language.Undetermined) : fast3CodeLookup.GetValueOrDefault(code, Language.Undetermined);
			if (l == Language.Undetermined)
				Console.WriteLine(lbl + "--" + code);

			return l;
		};
		
		Func<FastTextDetector, String, LanguagePrediction> pred = (fast, entry) => {
			// foreach (FastTextPrediction prediction in fast.Predict(entry, 3)) Console.WriteLine($"Language: {fast2CodeLookup.GetValueOrDefault(prediction.Label.Substring(9), Language.Undetermined)}, Probability: {prediction.Probability}");
			var first = fast.Predict(entry, 1).FirstOrDefault();
			if (first == null) return new LanguagePrediction(SpecialLanguageCodes.Undetermined, false, 0);
			
			return new LanguagePrediction(fastTextLabelToLanguage(first.Label), first.Probability > 0.25, first.Probability);
		};
		TestDetector("Fast176sm", () => {
			var detector = new FastTextDetector();
			detector.LoadDefaultModel();
			return detector;
		}, pred);
		
		TestDetector("Fast176", () => {
			var detector = new FastTextDetector();
			detector.LoadModel("data/lid.176.bin");
			return detector;
		}, pred);
		TestDetector("Fast217", () => {
			var detector = new FastTextDetector();
			detector.LoadModel("data/lid.218e.bin");
			return detector;
		}, pred);

		// Conflicts on fasttext.dll with Panlingo.fasttext
		// Func<FastTextWrapper, String, LanguagePrediction> predWrapper = (fast, entry) => {
		// 	Prediction predictSingle = fast.PredictSingle(entry);
		// 	return new(fastTextLabelToLanguage(predictSingle.Label), predictSingle.Probability > 0.25, predictSingle.Probability);
		// };
		// TestDetector("FastNetWrapper176", () => {
		// 	var detector = new FastTextWrapper();
		// 	detector.LoadModel("data/lid.176.bin");
		// 	return detector;
		// }, predWrapper);
		// TestDetector("FastNetWrapper217", () => {
		// 	var detector = new FastTextWrapper();
		// 	detector.LoadModel("data/lid.218e.bin");
		// 	return detector;
		// }, predWrapper);

		
		// TODO lingua.net 0.16.0
		
		// // Conflicts on lingua.dll with Panlingo.lingua
		// TestDetector("PioneerLingua-Low", () => {
		// 	var detector = Lingua.LanguageDetectorBuilder
		// 		.FromLanguages(Lingua.LanguageInfo.All().ToArray())
		// 		.WithPreloadedLanguageModels()
		// 		.WithLowAccuracyMode()
		// 		.Build();
		// 	return detector;
		// }, (detector, s) => {
		// 	var predictions = detector.ComputeLanguageConfidenceValues(s);
		// 	(Lingua.Language key, Double confidence) = predictions.First();
		// 	var isocode = Lingua.LanguageInfo.IsoCode6393(key).ToString().ToLowerInvariant();
		// 	return new LanguagePrediction(fast3CodeLookup[isocode], confidence > 0.25, confidence);
		// });
		//
		// TestDetector("PioneerLingua-High", () => {
		// 	var detector = Lingua.LanguageDetectorBuilder
		// 		.FromLanguages(Lingua.LanguageInfo.All().ToArray())
		// 		.WithPreloadedLanguageModels()
		// 		.Build();
		// 	return detector;
		// }, (detector, s) => {
		// 	var predictions = detector.ComputeLanguageConfidenceValues(s);
		// 	(Lingua.Language key, Double confidence) = predictions.First();
		// 	var isocode = Lingua.LanguageInfo.IsoCode6393(key).ToString().ToLowerInvariant();
		// 	return new LanguagePrediction(fast3CodeLookup[isocode], confidence > 0.25, confidence);
		// });

		Func<LinguaDetector,string,LanguagePrediction> linguaPred = (detector, s) => {
			IEnumerable<LinguaPrediction> linguaPredictions = detector.PredictLanguages(s);
			LinguaPrediction prediction = linguaPredictions.OrderByDescending(pred => pred.Confidence).FirstOrDefault();
			if (prediction == null) return new LanguagePrediction(Language.Undetermined, false, 0);
			Language lang = prediction.Language switch {
				LinguaLanguage.Afrikaans =>  Language.Afrikaans,
				LinguaLanguage.Albanian => Language.Albanian,
				LinguaLanguage.Arabic => Language.Arabic,
				LinguaLanguage.Armenian => Language.Armenian,
				LinguaLanguage.Azerbaijani => Language.Azerbaijani,
				LinguaLanguage.Basque => Language.Basque,
				LinguaLanguage.Belarusian => Language.Belarusian,
				LinguaLanguage.Bengali => Language.Bengali,
				LinguaLanguage.Bokmal => Language.Norwegian_Bokmal,
				LinguaLanguage.Bosnian => Language.Bosnian,
				LinguaLanguage.Bulgarian => Language.Bulgarian,
				LinguaLanguage.Catalan => Language.Catalan,
				LinguaLanguage.Chinese => Language.Chinese,
				LinguaLanguage.Croatian => Language.Croatian,
				LinguaLanguage.Czech => Language.Czech,
				LinguaLanguage.Danish => Language.Danish,
				LinguaLanguage.Dutch => Language.Dutch,
				LinguaLanguage.English => Language.English,
				LinguaLanguage.Esperanto => Language.Esperanto,
				LinguaLanguage.Estonian => Language.Estonian,
				LinguaLanguage.Finnish => Language.Finnish,
				LinguaLanguage.French => Language.French,
				LinguaLanguage.Ganda => Language.Ganda,
				LinguaLanguage.Georgian => Language.Georgian,
				LinguaLanguage.German => Language.German,
				LinguaLanguage.Greek => Language.Greek_Modern,
				LinguaLanguage.Gujarati => Language.Gujarati,
				LinguaLanguage.Hebrew => Language.Hebrew,
				LinguaLanguage.Hindi => Language.Hindi,
				LinguaLanguage.Hungarian => Language.Hungarian,
				LinguaLanguage.Icelandic => Language.Icelandic,
				LinguaLanguage.Indonesian => Language.Indonesian,
				LinguaLanguage.Irish => Language.Irish,
				LinguaLanguage.Italian => Language.Italian,
				LinguaLanguage.Japanese => Language.Japanese,
				LinguaLanguage.Kazakh => Language.Kazakh,
				LinguaLanguage.Korean => Language.Korean,
				LinguaLanguage.Latin => Language.Latin,
				LinguaLanguage.Latvian => Language.Latvian,
				LinguaLanguage.Lithuanian => Language.Lithuanian,
				LinguaLanguage.Macedonian => Language.Macedonian,
				LinguaLanguage.Malay => Language.Malay_macrolanguage,
				LinguaLanguage.Maori => Language.Maori,
				LinguaLanguage.Marathi => Language.Marathi,
				LinguaLanguage.Mongolian => Language.Mongolian,
				LinguaLanguage.Nynorsk => Language.Norwegian_Nynorsk,
				LinguaLanguage.Persian => Language.Persian,
				LinguaLanguage.Polish => Language.Polish,
				LinguaLanguage.Portuguese => Language.Portuguese,
				LinguaLanguage.Punjabi => Language.Panjabi,
				LinguaLanguage.Romanian => Language.Romanian,
				LinguaLanguage.Russian => Language.Russian,
				LinguaLanguage.Serbian => Language.Serbian,
				LinguaLanguage.Shona => Language.Shona,
				LinguaLanguage.Slovak => Language.Slovak,
				LinguaLanguage.Slovene => Language.Slovenian,
				LinguaLanguage.Somali => Language.Somali,
				LinguaLanguage.Sotho => Language.Sotho_Southern,
				LinguaLanguage.Spanish => Language.Spanish,
				LinguaLanguage.Swahili => Language.Swahili_macrolanguage,
				LinguaLanguage.Swedish => Language.Swedish,
				LinguaLanguage.Tagalog => Language.Tagalog,
				LinguaLanguage.Tamil => Language.Tamil,
				LinguaLanguage.Telugu => Language.Telugu,
				LinguaLanguage.Thai => Language.Thai,
				LinguaLanguage.Tsonga => Language.Tsonga,
				LinguaLanguage.Tswana => Language.Tswana,
				LinguaLanguage.Turkish => Language.Turkish,
				LinguaLanguage.Ukrainian => Language.Ukrainian,
				LinguaLanguage.Urdu => Language.Urdu,
				LinguaLanguage.Vietnamese => Language.Vietnamese,
				LinguaLanguage.Welsh => Language.Welsh,
				LinguaLanguage.Xhosa => Language.Xhosa,
				LinguaLanguage.Yoruba => Language.Yoruba,
				LinguaLanguage.Zulu => Language.Zulu,
				_ => Language.Undetermined,
			};
			return new(lang, prediction.Confidence > 0.25, prediction.Confidence);
		};
		TestDetector("Lingua-High", () => {
			using var linguaBuilder = new LinguaDetectorBuilder(Enum.GetValues<LinguaLanguage>())
					.WithPreloadedLanguageModels() // optional
				// .WithMinimumRelativeDistance(0.95) // optional
				// .WithLowAccuracyMode() // optional
				;
			return linguaBuilder.Build();
		}, linguaPred);
		
		TestDetector("Lingua-Low", () => {
			using var linguaBuilder = new LinguaDetectorBuilder(Enum.GetValues<LinguaLanguage>())
					.WithPreloadedLanguageModels() // optional
				// .WithMinimumRelativeDistance(0.95) // optional
				.WithLowAccuracyMode() // optional
				;
			return linguaBuilder.Build();
		}, linguaPred);
		
		TestDetector("Whatlang", () => new WhatlangDetector(), (detector, s) => {
			WhatlangPrediction? whatlangPrediction = detector.PredictLanguage(s);
			if (whatlangPrediction == null) return new LanguagePrediction(Language.Undetermined, false, 0);
			string languageCode = detector.GetLanguageCode(whatlangPrediction.Language);
			return new LanguagePrediction(fast3CodeLookup[languageCode], whatlangPrediction.IsReliable, whatlangPrediction.Confidence);
		});
	}

	public void TestDetector<T>(String name, Func<T> creator, Func<T, String, LanguagePrediction> detect) where T : class {
		PerformanceHelper.EstimateObjectSize(name, 1, 1, Console.Out, 0, _ => creator());
		_statistics[name] = new LanguageTestStats();
		Dictionary<Language, Int32> testLanguages = GetTestLanguages("single-words").ToDictionary(lang => lang, _ => 0);
		ConcurrentDictionary<Language, Int32> additionalLanguages = new();
		Stopwatch sw = new();
		T detector = creator();
		foreach (var category in new[] { "single-words", "word-pairs", "sentences" }) {
			LanguageTestStats perElementStat = new LanguageTestStats();
			foreach ((Language language, String[]? entries) in Get(category)) {
				LanguageTestStats perLanguageStat = new LanguageTestStats();
				sw.Start();
				foreach (string entry in entries) {
					LanguagePrediction languagePrediction = detect(detector, entry);
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
		Console.WriteLine($"Time for {_statistics[name].Predictions} predictions: {sw.Elapsed} or {sw.Elapsed.TotalMilliseconds/_statistics[name].Predictions:N3}ms per prediction");
		Console.WriteLine($"Unsupported Languages: {String.Join(", ",testLanguages.Where(kv => kv.Value == 0).Select(kv => kv.Key))}");
		Console.WriteLine($" Additional Languages: {String.Join(", ",additionalLanguages.Where(kv => kv.Value == 0).Select(kv => kv.Key))}");

		if (creator is IDisposable disposable) disposable.Dispose();
	}
}