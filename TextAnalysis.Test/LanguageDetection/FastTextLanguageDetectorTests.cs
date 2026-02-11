namespace TextAnalysis.Test.LanguageDetection;

using System.Collections.Frozen;
using IsoEnums.Iso639;
using TextAnalysis.LanguageDetection;

[TestFixture]
public class FastTextLanguageDetectorTests : ATest {
	[TestCase(null)]
	[TestCase(TestData.LanguageDetectionModels.FastText176)]
	[TestCase(TestData.LanguageDetectionModels.FastText218)]
	public void CanMapLabelsToLanguage(String? model) {
		using FastTextLanguageDetector detector = new(model);
		FrozenSet<Language> invalidLanguages = [Language.Uninitialized, Language.Undetermined, Language.No_linguistic_content];
		List<String> unmappedLanguageCodes = [];
		foreach (String fastTextLabel in detector.GetFastTextLabels()) {
			Language language = detector.MapFastTextLabel(fastTextLabel);
			Console.WriteLine($"{fastTextLabel} = {language} ({language.Get3Code()}, {language.Get2Code()})");
			if (invalidLanguages.Contains(language)) {
				unmappedLanguageCodes.Add(fastTextLabel);
			}
		}

		Console.WriteLine($"Unmapped codes: {unmappedLanguageCodes.Count}");
		unmappedLanguageCodes.ForEach(s => Console.WriteLine($"Unknown language code: {s}"));
		Assert.That(unmappedLanguageCodes, Is.Empty);
	}

	[Test]
	public void ThrowsOnModelNotFound() {
		Assert.That(() => {
			using FastTextLanguageDetector detector = new("");
		}, Throws.TypeOf<ArgumentException>());

		Assert.That(() => {
			using FastTextLanguageDetector detector = new("idonotexist.xxx");
		}, Throws.TypeOf<FileNotFoundException>());
	}
}