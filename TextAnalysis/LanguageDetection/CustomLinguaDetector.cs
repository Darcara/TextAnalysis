namespace TextAnalysis.LanguageDetection;

using System.Buffers;
using System.Collections.Frozen;
using System.Reflection;
using System.Runtime.InteropServices;
using IsoEnums.Iso639;
using Neco.Common;
using Neco.Common.Extensions;
using Panlingo.LanguageIdentification.Lingua;

// TODO custom native dll/so! Currently memory copy is needed to append \0 to the utf8-bytes so it becomes a zerio-terminated-cstring. This is unnecessary, as Rust can ust the utf8-ptr with length. 
internal sealed class CustomLinguaDetector : IDisposable {
	private readonly LinguaDetector _detector;
	private readonly LinguaDetectorBuilder _linguaBuilder;
	private readonly IntPtr _detectorPtr;
	private static readonly Int32 _structSize = _structSize = Marshal.SizeOf<LinguaPredictionResult>();

	private static readonly FrozenDictionary<LinguaLanguage, Language> _languageMap = _languageMap = new Dictionary<LinguaLanguage, Language> {
		{ LinguaLanguage.Afrikaans, Language.Afrikaans },
		{ LinguaLanguage.Albanian, Language.Albanian },
		{ LinguaLanguage.Arabic, Language.Arabic },
		{ LinguaLanguage.Armenian, Language.Armenian },
		{ LinguaLanguage.Azerbaijani, Language.Azerbaijani },
		{ LinguaLanguage.Basque, Language.Basque },
		{ LinguaLanguage.Belarusian, Language.Belarusian },
		{ LinguaLanguage.Bengali, Language.Bengali },
		{ LinguaLanguage.Bokmal, Language.Norwegian_Bokmal },
		{ LinguaLanguage.Bosnian, Language.Bosnian },
		{ LinguaLanguage.Bulgarian, Language.Bulgarian },
		{ LinguaLanguage.Catalan, Language.Catalan },
		{ LinguaLanguage.Chinese, Language.Chinese },
		{ LinguaLanguage.Croatian, Language.Croatian },
		{ LinguaLanguage.Czech, Language.Czech },
		{ LinguaLanguage.Danish, Language.Danish },
		{ LinguaLanguage.Dutch, Language.Dutch },
		{ LinguaLanguage.English, Language.English },
		{ LinguaLanguage.Esperanto, Language.Esperanto },
		{ LinguaLanguage.Estonian, Language.Estonian },
		{ LinguaLanguage.Finnish, Language.Finnish },
		{ LinguaLanguage.French, Language.French },
		{ LinguaLanguage.Ganda, Language.Ganda },
		{ LinguaLanguage.Georgian, Language.Georgian },
		{ LinguaLanguage.German, Language.German },
		{ LinguaLanguage.Greek, Language.Greek_Modern },
		{ LinguaLanguage.Gujarati, Language.Gujarati },
		{ LinguaLanguage.Hebrew, Language.Hebrew },
		{ LinguaLanguage.Hindi, Language.Hindi },
		{ LinguaLanguage.Hungarian, Language.Hungarian },
		{ LinguaLanguage.Icelandic, Language.Icelandic },
		{ LinguaLanguage.Indonesian, Language.Indonesian },
		{ LinguaLanguage.Irish, Language.Irish },
		{ LinguaLanguage.Italian, Language.Italian },
		{ LinguaLanguage.Japanese, Language.Japanese },
		{ LinguaLanguage.Kazakh, Language.Kazakh },
		{ LinguaLanguage.Korean, Language.Korean },
		{ LinguaLanguage.Latin, Language.Latin },
		{ LinguaLanguage.Latvian, Language.Latvian },
		{ LinguaLanguage.Lithuanian, Language.Lithuanian },
		{ LinguaLanguage.Macedonian, Language.Macedonian },
		{ LinguaLanguage.Malay, Language.Malay_macrolanguage },
		{ LinguaLanguage.Maori, Language.Maori },
		{ LinguaLanguage.Marathi, Language.Marathi },
		{ LinguaLanguage.Mongolian, Language.Mongolian },
		{ LinguaLanguage.Nynorsk, Language.Norwegian_Nynorsk },
		{ LinguaLanguage.Persian, Language.Persian },
		{ LinguaLanguage.Polish, Language.Polish },
		{ LinguaLanguage.Portuguese, Language.Portuguese },
		{ LinguaLanguage.Punjabi, Language.Panjabi },
		{ LinguaLanguage.Romanian, Language.Romanian },
		{ LinguaLanguage.Russian, Language.Russian },
		{ LinguaLanguage.Serbian, Language.Serbian },
		{ LinguaLanguage.Shona, Language.Shona },
		{ LinguaLanguage.Slovak, Language.Slovak },
		{ LinguaLanguage.Slovene, Language.Slovenian },
		{ LinguaLanguage.Somali, Language.Somali },
		{ LinguaLanguage.Sotho, Language.Sotho_Southern },
		{ LinguaLanguage.Spanish, Language.Spanish },
		{ LinguaLanguage.Swahili, Language.Swahili_macrolanguage },
		{ LinguaLanguage.Swedish, Language.Swedish },
		{ LinguaLanguage.Tagalog, Language.Tagalog },
		{ LinguaLanguage.Tamil, Language.Tamil },
		{ LinguaLanguage.Telugu, Language.Telugu },
		{ LinguaLanguage.Thai, Language.Thai },
		{ LinguaLanguage.Tsonga, Language.Tsonga },
		{ LinguaLanguage.Tswana, Language.Tswana },
		{ LinguaLanguage.Turkish, Language.Turkish },
		{ LinguaLanguage.Ukrainian, Language.Ukrainian },
		{ LinguaLanguage.Urdu, Language.Urdu },
		{ LinguaLanguage.Vietnamese, Language.Vietnamese },
		{ LinguaLanguage.Welsh, Language.Welsh },
		{ LinguaLanguage.Xhosa, Language.Xhosa },
		{ LinguaLanguage.Yoruba, Language.Yoruba },
		{ LinguaLanguage.Zulu, Language.Zulu },
	}.ToFrozenDictionary();


	public CustomLinguaDetector(Boolean useLowAccuracy = false) {
		_linguaBuilder = new LinguaDetectorBuilder(Enum.GetValues<LinguaLanguage>()).WithPreloadedLanguageModels();
		if (useLowAccuracy)
			_linguaBuilder.WithLowAccuracyMode();

		_detector = _linguaBuilder.Build();
		FieldInfo? fieldInfo = typeof(LinguaDetector).GetField("_detector", BindingFlags.Instance | BindingFlags.NonPublic);
		if (fieldInfo == null)
			throw new InvalidOperationException($"Unable to find detector in {typeof(LinguaDetector).GetFullName()}");
		_detectorPtr = (IntPtr)(fieldInfo.GetValue(_detector) ?? IntPtr.Zero);
	}

	public LanguagePrediction[] PredictLanguages(ReadOnlySpan<Char> text, Int32 count = 10) {
		Int32 maxBytes = MagicNumbers.Utf8NoBom.GetByteCount(text);
		Byte[] buffer = ArrayPool<Byte>.Shared.Rent(maxBytes + 1);
		Int32 utf8BytesUsed = MagicNumbers.Utf8NoBom.GetBytes(text, buffer);
		buffer[utf8BytesUsed] = 0;

		LanguagePrediction[] retval = PredictLanguagesCore(new ReadOnlySpan<Byte>(buffer, 0, utf8BytesUsed + 1), count);
		ArrayPool<Byte>.Shared.Return(buffer);
		return retval;
	}

	public LanguagePrediction[] PredictLanguages(ReadOnlySpan<Byte> utf8, Int32 count = 10) {
		Byte[] utf8Zt = ArrayPool<Byte>.Shared.Rent(utf8.Length + 1);
		utf8.CopyTo(utf8Zt);
		utf8Zt[utf8.Length + 1] = 0;
		LanguagePrediction[] retval = PredictLanguagesCore(new ReadOnlySpan<Byte>(utf8Zt, 0, utf8.Length + 1), count);
		ArrayPool<Byte>.Shared.Return(utf8Zt);
		return retval;
	}

	internal LanguagePrediction[] PredictLanguagesCore(ReadOnlySpan<Byte> utf8ZeroTerminated, Int32 count) {
		LinguaStatus status = LinguaDetectorWrapper.LinguaDetectSingle2(
			detector: _detectorPtr,
			utf8: utf8ZeroTerminated,
			result: out LinguaPredictionListResult nativeResult
		);

		return EvaluateLanguagePredictionResult(status, nativeResult, count);
	}

	internal LanguagePrediction[] PredictLanguages(String text, Int32 count) {
		LinguaStatus status = LinguaDetectorWrapper.LinguaDetectSingle(
			detector: _detectorPtr,
			text: text,
			result: out LinguaPredictionListResult nativeResult
		);

		return EvaluateLanguagePredictionResult(status, nativeResult, count);
	}
	
	private LanguagePrediction[] EvaluateLanguagePredictionResult(LinguaStatus status, LinguaPredictionListResult nativeResult, Int32 count) {
		try {
			if (status == LinguaStatus.DetectFailure)
				return [];

			if (status == LinguaStatus.BadTextPtr || status == LinguaStatus.BadOutputPtr)
				throw new LinguaDetectorException($"Failed to detect language: {status}");

			Int32 toTake = Math.Min(count, nativeResult.PredictionsCount);
			LanguagePrediction[] retval = new LanguagePrediction[toTake];
			
			for (Int32 i = 0; i < toTake; i++) {
				LinguaPredictionResult prediction = Marshal.PtrToStructure<LinguaPredictionResult>(nativeResult.Predictions + i * _structSize);
				retval[i] = new LanguagePrediction(_languageMap.GetValueOrDefault(prediction.Language, Language.Undetermined), prediction.Confidence > 0.25, prediction.Confidence);
			}

			return retval;
		}
		finally {
			LinguaDetectorWrapper.LinguaPredictionResultDestroy(nativeResult.Predictions);
		}
	}

	#region IDisposable

	/// <inheritdoc />
	public void Dispose() {
		_detector.Dispose();
		_linguaBuilder.Dispose();
	}

	#endregion
}