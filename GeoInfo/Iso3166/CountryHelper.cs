namespace GeoInfo.Iso3166;

using System.Collections.Frozen;
using GeoInfo.Iso639;

public static class CountryHelper {
	public const String Unavailable2 = "??";
	internal static ReadOnlySpan<Byte> Unavailable2Bytes => "??"u8;
	public const String Unavailable3 = "???";
	internal static ReadOnlySpan<Byte> Unavailable3Bytes => "???"u8;

	public static FrozenDictionary<String, Country> CreateFast2CodeLookup() => Enum.GetValues<Country>().Where(l => l.Has2Code()).ToFrozenDictionary(l => l.Get2Code(), l => l, StringComparer.OrdinalIgnoreCase);
	public static FrozenDictionary<String, Country> CreateFast3CodeLookup() => Enum.GetValues<Country>().Where(l => l > Country.Uninitialized).ToFrozenDictionary(l => l.Get3Code(), l => l, StringComparer.OrdinalIgnoreCase);
	
	public static Country GetCountryByCode(ReadOnlySpan<Char> countryCode) {
		if (countryCode.Length < 2) return Country.NotACountry;
		if (countryCode.Length == 2) return GetCountryBy2Code(countryCode);
		if (countryCode.Length == 3) return GetCountryBy3Code(countryCode);

		if (countryCode[2] == '-' || countryCode[2] == '_') return GetCountryBy2Code(countryCode.Slice(0, 2));
		if (countryCode.Length > 3 && (countryCode[3] == '-' || countryCode[3] == '_')) return GetCountryBy3Code(countryCode.Slice(0, 3));
		return Country.NotACountry;
	}
	
	public static Country GetCountryByCode(ReadOnlySpan<Byte> countryCode) {
		if (countryCode.Length < 2) return Country.NotACountry;
		if (countryCode.Length == 2) return GetCountryBy2Code(countryCode);
		if (countryCode.Length == 3) return GetCountryBy3Code(countryCode);

		if (countryCode[2] == '-' || countryCode[2] == '_') return GetCountryBy2Code(countryCode.Slice(0, 2));
		if (countryCode.Length > 3 && (countryCode[3] == '-' || countryCode[3] == '_')) return GetCountryBy3Code(countryCode.Slice(0, 3));
		return Country.NotACountry;
	}


	/// <summary>
	/// 
	/// </summary>
	/// <param name="country2Code">ISO 3166 country code (2 letters)</param>
	/// <returns></returns>
	public static Country GetCountryBy2Code(ReadOnlySpan<Char> country2Code) {
		if(country2Code.Length != 2 || country2Code.SequenceEqual(Unavailable2)) return Country.NotACountry;
		Span<Char> lower = stackalloc Char[2];
		country2Code.ToLowerInvariant(lower);
		foreach (Country country in Enum.GetValues<Country>()) {
			if(lower.SequenceEqual(country.Get2Code())) return country;
		}
		return Country.NotACountry;
	}
	
	public static Country GetCountryBy2Code(ReadOnlySpan<Byte> country2Code) {
		if(country2Code.Length != 2 || country2Code.SequenceEqual(Unavailable2Bytes)) return Country.NotACountry;
		Span<Byte> buffer = stackalloc Byte[4];
		Span<Byte> lower = buffer.Slice(0, 2);
		lower[0] = (Byte)(country2Code[0] < 97 ? country2Code[0] + 32 : country2Code[0]);
		lower[1] = (Byte)(country2Code[1] < 97 ? country2Code[1] + 32 : country2Code[1]);
		Span<Byte> codeBuffer = buffer.Slice( 2);
		foreach (Country country in Enum.GetValues<Country>()) {
			country.Get2CodeBytes(codeBuffer);
			if(lower.SequenceEqual(codeBuffer)) return country;
		}
		return Country.NotACountry;
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="country3Code">ISO 3166 country code (3 letters)</param>
	/// <returns></returns>
	public static Country GetCountryBy3Code(ReadOnlySpan<Char> country3Code) {
		if(country3Code.Length != 3|| country3Code.SequenceEqual(Unavailable3)) return Country.NotACountry;

		Span<Char> lower = stackalloc Char[3];
		country3Code.ToLowerInvariant(lower);
		foreach (Country country in Enum.GetValues<Country>()) {
			if(lower.SequenceEqual(country.Get3Code())) return country;
		}
		return Country.NotACountry;
	}
	
	public static Country GetCountryBy3Code(ReadOnlySpan<Byte> country3Code) {
		if(country3Code.Length != 3 || country3Code.SequenceEqual(Unavailable3Bytes)) return Country.NotACountry;
		Span<Byte> buffer = stackalloc Byte[6];
		Span<Byte> lower = buffer.Slice(0, 3);
		lower[0] = (Byte)(country3Code[0] < 97 ? country3Code[0] + 32 : country3Code[0]);
		lower[1] = (Byte)(country3Code[1] < 97 ? country3Code[1] + 32 : country3Code[1]);
		lower[2] = (Byte)(country3Code[2] < 97 ? country3Code[2] + 32 : country3Code[2]);
		Span<Byte> codeBuffer = buffer.Slice( 3);
		foreach (Country country in Enum.GetValues<Country>()) {
			country.Get3CodeBytes(codeBuffer);
			if(lower.SequenceEqual(codeBuffer)) return country;
		}
		return Country.NotACountry;
	}
}