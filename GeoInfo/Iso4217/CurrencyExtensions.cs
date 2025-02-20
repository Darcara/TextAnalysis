namespace GeoInfo.Iso4217;

public static class CurrencyExtensions {
	public static String Get3Code(this Currency currency) {
		if (!Enum.IsDefined(currency) || currency <= Currency.Uninitialized) return CurrencyHelper.Unavailable3;
		Int32 value = (Int32)currency - 1;
		Span<Char> chars = stackalloc Char[3];
		chars[0] = (Char)((Byte)'a' + ((value >> 1) & 0b11111));
		chars[1] = (Char)((Byte)'a' + ((value >> 6) & 0b11111));
		chars[2] = (Char)((Byte)'a' + ((value >> 11) & 0b11111));
		return new String(chars);
	}

	public static void Get3CodeBytes(this Currency currency, Span<Byte> bytes) {
		if (!Enum.IsDefined(currency) || currency <= Currency.Uninitialized) {
			CurrencyHelper.Unavailable3Bytes.CopyTo(bytes);
			return;
		}

		Int32 value = (Int32)currency - 1;
		bytes[0] = (Byte)((Byte)'a' + ((value >> 1) & 0b11111));
		bytes[1] = (Byte)((Byte)'a' + ((value >> 6) & 0b11111));
		bytes[2] = (Byte)((Byte)'a' + ((value >> 11) & 0b11111));
	}
}