namespace TextAnalysis.Test.GeoInfo;

using global::GeoInfo.Iso4217;
using Neco.Common.Helper;

[TestFixture]
public class CurrencyTests {
	[TestCase(Currency.Afghani, "afn")]
	[TestCase(Currency.Euro, "eur")]
	[TestCase(Currency.Loti, "lsl")]
	[TestCase(Currency.SomaliShilling, "sos")]
	[TestCase(-2, CurrencyHelper.Unavailable3)]
	[TestCase(Currency.Uninitialized, CurrencyHelper.Unavailable3)]
	[TestCase(Currency.NotACurrency, CurrencyHelper.Unavailable3)]
	public void CanGet3Code(Currency currency, String code) {
		currency.Get3Code().Should().Be(code);
		if (code == CurrencyHelper.Unavailable3) {
			CurrencyHelper.GetCurrencyBy3Code(code).Should().Be(Currency.NotACurrency);
		} else {
			CurrencyHelper.CreateFast3CodeLookup()[code].Should().Be(currency);
			CurrencyHelper.GetCurrencyBy3Code(code).Should().Be(currency);
		}
	}
	
	[TestCase("afn", Currency.Afghani)]
	[TestCase("AFN", Currency.Afghani)]
	[TestCase("AfN", Currency.Afghani)]
	[TestCase("afn-whatever", Currency.NotACurrency)]
	[TestCase("AFN-whatever", Currency.NotACurrency)]
	[TestCase("eur", Currency.Euro)]
	[TestCase("EUR", Currency.Euro)]
	[TestCase("X", Currency.NotACurrency)]
	[TestCase("XXXX", Currency.NotACurrency)]
	[TestCase("", Currency.NotACurrency)]
	[TestCase(CurrencyHelper.Unavailable3, Currency.NotACurrency)]
	public void CanGetByCodeBytes(String code, Currency language) {
		Byte[] bytes = new Byte[code.Length];
		for (var i = 0; i < code.Length; i++) {
			bytes[i] = (Byte)code[i];
		}
		CurrencyHelper.GetCurrencyBy3Code(bytes).Should().Be(language);
	}

	[Category("Benchmark")]
	[Test]
	public void CanCreateFastLookups() {
		PerformanceHelper.EstimateObjectSize("3CodeLookup", 0, _ => CurrencyHelper.CreateFast3CodeLookup());
	}

	[Test]
	public void ValuesAreUnique() {
		Currency[] languages = Enum.GetValues<Currency>();
		languages.Select(l => (Int32)l).Distinct().Should().HaveCount(languages.Length);
	}
}