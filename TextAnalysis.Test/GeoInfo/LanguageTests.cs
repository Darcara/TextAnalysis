namespace TextAnalysis.Test.GeoInfo;

using global::GeoInfo.Iso639;
using Neco.Common.Helper;

[TestFixture]
public class LanguageTests {
	[TestCase(Language.Afar, "aa")]
	[TestCase(Language.Ghotuo, LanguageHelper.Unavailable2)]
	[TestCase(Language.Abkhazian, "ab")]
	[TestCase(Language.Afrikaans, "af")]
	[TestCase(Language.Armenian, "hy")]
	[TestCase(-1, LanguageHelper.Unavailable2)]
	[TestCase(Language.Uninitialized, LanguageHelper.Unavailable2)]
	public void CanGet2Code(Language language, String code) {
		language.Get2Code().Should().Be(code);
		if (code == LanguageHelper.Unavailable2) {
			LanguageHelper.GetLanguageBy2Code(code).Should().Be(Language.Undetermined);
			LanguageHelper.GetLanguageByCode(code).Should().Be(Language.Undetermined);
		} else {
			LanguageHelper.CreateFast2CodeLookup()[code].Should().Be(language);
			LanguageHelper.GetLanguageBy2Code(code).Should().Be(language);
			LanguageHelper.GetLanguageByCode(code).Should().Be(language);
		}
	}

	[TestCase(Language.Afar, "aar")]
	[TestCase(Language.Ghotuo, "aaa")]
	[TestCase(Language.Abkhazian, "abk")]
	[TestCase(Language.Afrikaans, "afr")]
	[TestCase(Language.Armenian, "hye")]
	[TestCase(-1, LanguageHelper.Unavailable3)]
	[TestCase(Language.Uninitialized, LanguageHelper.Unavailable3)]
	public void CanGet3Code(Language language, String code) {
		language.Get3Code().Should().Be(code);
		if (code == LanguageHelper.Unavailable3) {
			LanguageHelper.GetLanguageBy3Code(code).Should().Be(Language.Undetermined);
			LanguageHelper.GetLanguageByCode(code).Should().Be(Language.Undetermined);
		} else {
			LanguageHelper.CreateFast3CodeLookup()[code].Should().Be(language);
			LanguageHelper.GetLanguageBy3Code(code).Should().Be(language);
			LanguageHelper.GetLanguageByCode(code).Should().Be(language);
		}
	}

	[TestCase("hy", Language.Armenian)]
	[TestCase("HY", Language.Armenian)]
	[TestCase("hy-whatever", Language.Armenian)]
	[TestCase("HY-whatever", Language.Armenian)]
	[TestCase("hye", Language.Armenian)]
	[TestCase("HYE", Language.Armenian)]
	[TestCase("X", Language.Undetermined)]
	[TestCase("XXXX", Language.Undetermined)]
	[TestCase("", Language.Undetermined)]
	[TestCase(LanguageHelper.Unavailable2, Language.Undetermined)]
	[TestCase(LanguageHelper.Unavailable3, Language.Undetermined)]
	public void CanGetByCode(String code, Language language) {
		LanguageHelper.GetLanguageByCode(code).Should().Be(language);
	}
	
	[TestCase("hy", Language.Armenian)]
	[TestCase("HY", Language.Armenian)]
	[TestCase("hy-whatever", Language.Armenian)]
	[TestCase("HY-whatever", Language.Armenian)]
	[TestCase("hye", Language.Armenian)]
	[TestCase("HYE", Language.Armenian)]
	[TestCase("X", Language.Undetermined)]
	[TestCase("XXXX", Language.Undetermined)]
	[TestCase("", Language.Undetermined)]
	[TestCase(LanguageHelper.Unavailable2, Language.Undetermined)]
	[TestCase(LanguageHelper.Unavailable3, Language.Undetermined)]
	public void CanGetByCodeBytes(String code, Language language) {
		Byte[] bytes = new Byte[code.Length];
		for (var i = 0; i < code.Length; i++) {
			bytes[i] = (Byte)code[i];
		}
		LanguageHelper.GetLanguageByCode(bytes).Should().Be(language);
	}

	[Category("Benchmark")]
	[Test]
	public void CanCreateFastLookups() {
		PerformanceHelper.EstimateObjectSize("2CodeLookup",0, _ => LanguageHelper.CreateFast2CodeLookup());
		PerformanceHelper.EstimateObjectSize("3CodeLookup", 0, _ => LanguageHelper.CreateFast3CodeLookup());
	}

	[Test]
	public void ValuesAreUnique() {
		Language[] languages = Enum.GetValues<Language>();
		languages.Select(l => (Int32)l).Distinct().Should().HaveCount(languages.Length);
	}
}