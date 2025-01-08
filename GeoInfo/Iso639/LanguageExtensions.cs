namespace GeoInfo.Iso639;

public static class LanguageExtensions {


	public static Boolean Has2Code(this Language language) {
		if (!Enum.IsDefined(language) || language == Language.Uninitialized) return false;
		UInt32 value = ((UInt32)language)&0xFFFF0000;
		if (value == 0x0000) return false;
		return true;
	}

	public static String Get2Code(this Language language) {
		if (!Enum.IsDefined(language) || language == Language.Uninitialized) return LanguageHelper.Unavailable2;
		UInt32 value = ((UInt32)language)&0xFFFF0000;
		if (value == 0x0000) return LanguageHelper.Unavailable2;
		--value;
		Span<Char> chars = stackalloc Char[2];
		chars[0] = (Char)((Byte)'a' + ((value >> 17) & 0b11111));
		chars[1] = (Char)((Byte)'a' + ((value >> 22) & 0b11111));
		return new String(chars);
	}

	public static String Get3Code(this Language language) {
		if (!Enum.IsDefined(language) || language == Language.Uninitialized) return LanguageHelper.Unavailable3;
		Int32 value = (Int32)language -1;
		Span<Char> chars = stackalloc Char[3];
		chars[0] = (Char)((Byte)'a' + ((value >> 1) & 0b11111));
		chars[1] = (Char)((Byte)'a' + ((value >> 6) & 0b11111));
		chars[2] = (Char)((Byte)'a' + ((value >> 11) & 0b11111));
		return new String(chars);
	}
}