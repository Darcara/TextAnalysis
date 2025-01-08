namespace GeoInfo.Iso639;

using System.Collections.Frozen;

public static class LanguageHelper {
	public const String Unavailable2 = "??";
	public const String Unavailable3 = "???";

	public static FrozenDictionary<String, Language> CreateFast2CodeLookup() => Enum.GetValues<Language>().Where(l => l.Has2Code()).ToFrozenDictionary(l => l.Get2Code(), l => l, StringComparer.OrdinalIgnoreCase);
	public static FrozenDictionary<String, Language> CreateFast3CodeLookup() => Enum.GetValues<Language>().Where(l => l > Language.Uninitialized).ToFrozenDictionary(l => l.Get3Code(), l => l, StringComparer.OrdinalIgnoreCase);

	public static Language GetLanguageByCode(ReadOnlySpan<Char> languageCode) {
		if (languageCode.Length < 2) return Language.Undetermined;
		if (languageCode.Length == 2) return GetLanguageBy2Code(languageCode);
		if (languageCode.Length == 3) return GetLanguageBy3Code(languageCode);

		if (languageCode[2] == '-' || languageCode[2] == '_') return GetLanguageBy2Code(languageCode.Slice(0, 2));
		return Language.Undetermined;
	}

	public static Language GetLanguageBy2Code(ReadOnlySpan<Char> language2Code) {
		if(language2Code.Length != 2 || language2Code.SequenceEqual(Unavailable2)) return Language.Undetermined;
		Span<Char> lower = stackalloc Char[2];
		language2Code.ToLowerInvariant(lower);
		foreach (Language language in Enum.GetValues<Language>()) {
			if(lower.SequenceEqual(language.Get2Code())) return language;
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
}