namespace GeoInfo.Iso4217;

using System.Collections.Frozen;

public static class CurrencyHelper {
	public const String Unavailable3 = "???";
	internal static ReadOnlySpan<Byte> Unavailable3Bytes => "???"u8;
	public static FrozenDictionary<String, Currency> CreateFast3CodeLookup() => Enum.GetValues<Currency>().Where(l => l > Currency.Uninitialized).ToFrozenDictionary(l => l.Get3Code(), l => l, StringComparer.OrdinalIgnoreCase);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="currency3Code">ISO 4217 currency code (3 letters)</param>
	/// <returns></returns>
	public static Currency GetCurrencyBy3Code(ReadOnlySpan<Char> currency3Code) {
		if (currency3Code.Length != 3 || currency3Code.SequenceEqual(Unavailable3)) return Currency.NotACurrency;

		Span<Char> lower = stackalloc Char[3];
		currency3Code.ToLowerInvariant(lower);
		foreach (Currency currency in Enum.GetValues<Currency>()) {
			if (lower.SequenceEqual(currency.Get3Code())) return currency;
		}

		return Currency.NotACurrency;
	}

	public static Currency GetCurrencyBy3Code(ReadOnlySpan<Byte> currency3Code) {
		if (currency3Code.Length != 3 || currency3Code.SequenceEqual(Unavailable3Bytes)) return Currency.NotACurrency;
		Span<Byte> buffer = stackalloc Byte[6];
		Span<Byte> lower = buffer.Slice(0, 3);
		lower[0] = (Byte)(currency3Code[0] < 97 ? currency3Code[0] + 32 : currency3Code[0]);
		lower[1] = (Byte)(currency3Code[1] < 97 ? currency3Code[1] + 32 : currency3Code[1]);
		lower[2] = (Byte)(currency3Code[2] < 97 ? currency3Code[2] + 32 : currency3Code[2]);
		Span<Byte> codeBuffer = buffer.Slice(3);
		foreach (Currency currency in Enum.GetValues<Currency>()) {
			currency.Get3CodeBytes(codeBuffer);
			if (lower.SequenceEqual(codeBuffer)) return currency;
		}

		return Currency.NotACurrency;
	}
}