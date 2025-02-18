namespace GeoInfo.Iso3166;

public static class CountryExtensions {
	public static Boolean Has2Code(this Country country) {
		if (!Enum.IsDefined(country) || country == Country.Uninitialized) return false;
		UInt32 value = ((UInt32)country) & 0xFFFF0000;
		if (value == 0x0000) return false;
		return true;
	}

	public static String Get2Code(this Country country) {
		if (!Enum.IsDefined(country) || country <= Country.Uninitialized) return CountryHelper.Unavailable2;
		UInt32 value = ((UInt32)country) & 0xFFFF0000;
		if (value == 0x0000) return CountryHelper.Unavailable2;
		--value;
		Span<Char> chars = stackalloc Char[2];
		chars[0] = (Char)((Byte)'a' + ((value >> 17) & 0b11111));
		chars[1] = (Char)((Byte)'a' + ((value >> 22) & 0b11111));
		return new String(chars);
	}

	internal static void Get2CodeBytes(this Country country, Span<Byte> bytes) {
		if (!Enum.IsDefined(country) || country <= Country.Uninitialized) {
			CountryHelper.Unavailable2Bytes.CopyTo(bytes);
			return;
		}

		UInt32 value = ((UInt32)country) & 0xFFFF0000;
		if (value == 0x0000) {
			CountryHelper.Unavailable2Bytes.CopyTo(bytes);
			return;
		}

		--value;
		bytes[0] = (Byte)((Byte)'a' + ((value >> 17) & 0b11111));
		bytes[1] = (Byte)((Byte)'a' + ((value >> 22) & 0b11111));
	}

	public static String Get3Code(this Country country) {
		if (!Enum.IsDefined(country) || country <= Country.Uninitialized) return CountryHelper.Unavailable3;
		Int32 value = (Int32)country - 1;
		Span<Char> chars = stackalloc Char[3];
		chars[0] = (Char)((Byte)'a' + ((value >> 1) & 0b11111));
		chars[1] = (Char)((Byte)'a' + ((value >> 6) & 0b11111));
		chars[2] = (Char)((Byte)'a' + ((value >> 11) & 0b11111));
		return new String(chars);
	}

	public static void Get3CodeBytes(this Country country, Span<Byte> bytes) {
		if (!Enum.IsDefined(country) || country <= Country.Uninitialized) {
			CountryHelper.Unavailable3Bytes.CopyTo(bytes);
			return;
		}

		Int32 value = (Int32)country - 1;
		bytes[0] = (Byte)((Byte)'a' + ((value >> 1) & 0b11111));
		bytes[1] = (Byte)((Byte)'a' + ((value >> 6) & 0b11111));
		bytes[2] = (Byte)((Byte)'a' + ((value >> 11) & 0b11111));
	}
}