namespace TextAnalysis.Test.GeoInfo;

using global::GeoInfo.Iso3166;
using Neco.Common.Helper;

[TestFixture]
public class CountryTests {
	[TestCase(Country.Afghanistan, "af")]
	[TestCase(Country.Canada, "ca")]
	[TestCase(Country.Gabon, "ga")]
	[TestCase(Country.BosniaandHerzegovina, "ba")]
	[TestCase(-2, CountryHelper.Unavailable2)]
	[TestCase(Country.Uninitialized, CountryHelper.Unavailable2)]
	[TestCase(Country.NotACountry, CountryHelper.Unavailable2)]
	public void CanGet2Code(Country country, String code) {
		country.Get2Code().Should().Be(code);
		if (code == CountryHelper.Unavailable2) {
			CountryHelper.GetCountryBy2Code(code).Should().Be(Country.NotACountry);
			CountryHelper.GetCountryByCode(code).Should().Be(Country.NotACountry);
		} else {
			CountryHelper.CreateFast2CodeLookup()[code].Should().Be(country);
			CountryHelper.GetCountryBy2Code(code).Should().Be(country);
			CountryHelper.GetCountryByCode(code).Should().Be(country);
		}
	}
	
	[TestCase(Country.Afghanistan, "afg")]
	[TestCase(Country.Canada, "can")]
	[TestCase(Country.Gabon, "gab")]
	[TestCase(Country.BosniaandHerzegovina, "bih")]
	[TestCase(-2, CountryHelper.Unavailable3)]
	[TestCase(Country.Uninitialized, CountryHelper.Unavailable3)]
	[TestCase(Country.NotACountry, CountryHelper.Unavailable3)]
	public void CanGet3Code(Country country, String code) {
		country.Get3Code().Should().Be(code);
		if (code == CountryHelper.Unavailable3) {
			CountryHelper.GetCountryBy3Code(code).Should().Be(Country.NotACountry);
			CountryHelper.GetCountryByCode(code).Should().Be(Country.NotACountry);
		} else {
			CountryHelper.CreateFast3CodeLookup()[code].Should().Be(country);
			CountryHelper.GetCountryBy3Code(code).Should().Be(country);
			CountryHelper.GetCountryByCode(code).Should().Be(country);
		}
	}
	
	[TestCase("ca", Country.Canada)]
	[TestCase("CA", Country.Canada)]
	[TestCase("ca-whatever", Country.Canada)]
	[TestCase("CA-whatever", Country.Canada)]
	[TestCase("ca_whatever", Country.Canada)]
	[TestCase("CA_whatever", Country.Canada)]
	[TestCase("can-whatever", Country.Canada)]
	[TestCase("CAN-whatever", Country.Canada)]
	[TestCase("can_whatever", Country.Canada)]
	[TestCase("CAN_whatever", Country.Canada)]
	[TestCase("can", Country.Canada)]
	[TestCase("CAN", Country.Canada)]
	[TestCase("X", Country.NotACountry)]
	[TestCase("XXXX", Country.NotACountry)]
	[TestCase("", Country.NotACountry)]
	[TestCase(CountryHelper.Unavailable2, Country.NotACountry)]
	[TestCase(CountryHelper.Unavailable3, Country.NotACountry)]
	public void CanGetByCodeBytes(String code, Country language) {
		Byte[] bytes = new Byte[code.Length];
		for (var i = 0; i < code.Length; i++) {
			bytes[i] = (Byte)code[i];
		}
		CountryHelper.GetCountryByCode(bytes).Should().Be(language);
		CountryHelper.GetCountryByCode(code).Should().Be(language);
	}

	[Category("Benchmark")]
	[Test]
	public void CanCreateFastLookups() {
		PerformanceHelper.EstimateObjectSize("2CodeLookup",0, _ => CountryHelper.CreateFast2CodeLookup());
		PerformanceHelper.EstimateObjectSize("3CodeLookup", 0, _ => CountryHelper.CreateFast3CodeLookup());
	}

	[Test]
	public void ValuesAreUnique() {
		Country[] languages = Enum.GetValues<Country>();
		languages.Select(l => (Int32)l).Distinct().Should().HaveCount(languages.Length);
	}
}