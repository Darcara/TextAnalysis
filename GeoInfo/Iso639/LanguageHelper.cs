namespace GeoInfo.Iso639;

using System.Collections.Frozen;

public static class LanguageHelper {
	public const String Unavailable2 = "??";
	internal static ReadOnlySpan<Byte> Unavailable2Bytes => "??"u8;
	public const String Unavailable3 = "???";
	internal static ReadOnlySpan<Byte> Unavailable3Bytes => "???"u8;

	public static FrozenDictionary<String, Language> CreateFast2CodeLookup() => Enum.GetValues<Language>().Where(l => l.Has2Code()).ToFrozenDictionary(l => l.Get2Code(), l => l, StringComparer.OrdinalIgnoreCase);
	public static FrozenDictionary<String, Language> CreateFast3CodeLookup() => Enum.GetValues<Language>().Where(l => l > Language.Uninitialized).ToFrozenDictionary(l => l.Get3Code(), l => l, StringComparer.OrdinalIgnoreCase);
	
	public static Language GetLanguageByCode(ReadOnlySpan<Char> languageCode) {
		if (languageCode.Length < 2) return Language.Undetermined;
		if (languageCode.Length == 2) return GetLanguageBy2Code(languageCode);
		if (languageCode.Length == 3) return GetLanguageBy3Code(languageCode);

		if (languageCode[2] == '-' || languageCode[2] == '_') return GetLanguageBy2Code(languageCode.Slice(0, 2));
		if (languageCode.Length > 3 && (languageCode[3] == '-' || languageCode[3] == '_')) return GetLanguageBy3Code(languageCode.Slice(0, 3));
		return Language.Undetermined;
	}
	
	public static Language GetLanguageByCode(ReadOnlySpan<Byte> languageCode) {
		if (languageCode.Length < 2) return Language.Undetermined;
		if (languageCode.Length == 2) return GetLanguageBy2Code(languageCode);
		if (languageCode.Length == 3) return GetLanguageBy3Code(languageCode);

		if (languageCode[2] == '-' || languageCode[2] == '_') return GetLanguageBy2Code(languageCode.Slice(0, 2));
		if (languageCode.Length > 3 && (languageCode[3] == '-' || languageCode[3] == '_')) return GetLanguageBy3Code(languageCode.Slice(0, 3));
		return Language.Undetermined;
	}


	/// <summary>
	/// 
	/// </summary>
	/// <param name="language2Code">ISO 639-1 language code (2 letters)</param>
	/// <returns></returns>
	public static Language GetLanguageBy2Code(ReadOnlySpan<Char> language2Code) {
		if(language2Code.Length != 2 || language2Code.SequenceEqual(Unavailable2)) return Language.Undetermined;
		Span<Char> lower = stackalloc Char[2];
		language2Code.ToLowerInvariant(lower);
		foreach (Language language in Enum.GetValues<Language>()) {
			if(lower.SequenceEqual(language.Get2Code())) return language;
		}
		return Language.Undetermined;
	}
	
	public static Language GetLanguageBy2Code(ReadOnlySpan<Byte> language2Code) {
		if(language2Code.Length != 2 || language2Code.SequenceEqual(Unavailable2Bytes)) return Language.Undetermined;
		Span<Byte> buffer = stackalloc Byte[4];
		Span<Byte> lower = buffer.Slice(0, 2);
		lower[0] = (Byte)(language2Code[0] < 97 ? language2Code[0] + 32 : language2Code[0]);
		lower[1] = (Byte)(language2Code[1] < 97 ? language2Code[1] + 32 : language2Code[1]);
		Span<Byte> codeBuffer = buffer.Slice( 2);
		foreach (Language language in Enum.GetValues<Language>()) {
			language.Get2CodeBytes(codeBuffer);
			if(lower.SequenceEqual(codeBuffer)) return language;
		}
		return Language.Undetermined;
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="language3Code">ISO 639-3 language code (3 letters)</param>
	/// <returns></returns>
	public static Language GetLanguageBy3Code(ReadOnlySpan<Char> language3Code) {
		if(language3Code.Length != 3|| language3Code.SequenceEqual(Unavailable3)) return Language.Undetermined;

		Span<Char> lower = stackalloc Char[3];
		language3Code.ToLowerInvariant(lower);
		foreach (Language language in Enum.GetValues<Language>()) {
			if(lower.SequenceEqual(language.Get3Code())) return language;
		}
		return Language.Undetermined;
	}
	
	public static Language GetLanguageBy3Code(ReadOnlySpan<Byte> language3Code) {
		if(language3Code.Length != 3 || language3Code.SequenceEqual(Unavailable3Bytes)) return Language.Undetermined;
		Span<Byte> buffer = stackalloc Byte[6];
		Span<Byte> lower = buffer.Slice(0, 3);
		lower[0] = (Byte)(language3Code[0] < 97 ? language3Code[0] + 32 : language3Code[0]);
		lower[1] = (Byte)(language3Code[1] < 97 ? language3Code[1] + 32 : language3Code[1]);
		lower[2] = (Byte)(language3Code[2] < 97 ? language3Code[2] + 32 : language3Code[2]);
		Span<Byte> codeBuffer = buffer.Slice( 3);
		foreach (Language language in Enum.GetValues<Language>()) {
			language.Get3CodeBytes(codeBuffer);
			if(lower.SequenceEqual(codeBuffer)) return language;
		}
		return Language.Undetermined;
	}
}