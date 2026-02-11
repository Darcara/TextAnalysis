namespace TextAnalysis.Benchmark;

using IsoEnums.Iso639;
using Panlingo.LanguageIdentification.Lingua;
using TextAnalysis.LanguageDetection;

public class LinguaDetection {
	private CustomLinguaDetector _lingua = null!;
	private LinguaDetector _linguaOriginal = null!;
	private Byte[] _utf8Zt = null!;
	private const String _sentenceUkr = "Запропоновано і впроваджено комплекс індивідуальних психокорекційних вправ та групової корекції, які спрямовано на підвищення можливостей самоактуалізації, поліпшення комунікативної, поведінкової та емоційно-вольової сфери в осіб з невротичними розладами.";
	private const Int32 _sentenceUkrSplitIndex = 95; // the first comma
	private const Int32 _sentenceUkrBytesSplitIndex = 182; // the first comma
	private ReadOnlySpan<Byte> SentenceUkrBytes => "Запропоновано і впроваджено комплекс індивідуальних психокорекційних вправ та групової корекції, які спрямовано на підвищення можливостей самоактуалізації, поліпшення комунікативної, поведінкової та емоційно-вольової сфери в осіб з невротичними розладами."u8;

	[Params(3, 10)] 
	public Int32 NumLang { get; set; }

	[GlobalSetup]
	public void Setup() {
		_lingua = new CustomLinguaDetector();
		
		var linguaBuilder = new LinguaDetectorBuilder(Enum.GetValues<LinguaLanguage>()).WithPreloadedLanguageModels();
		_linguaOriginal = linguaBuilder.Build();
		
		_utf8Zt = new Byte[_sentenceUkrBytesSplitIndex + 1];
		SentenceUkrBytes.Slice(0, _sentenceUkrBytesSplitIndex).CopyTo(_utf8Zt);
		_utf8Zt[_sentenceUkrBytesSplitIndex] = 0;
	}

	[Benchmark(Baseline = true)]
	public LinguaLanguage PanlingoDefault() {
		return _linguaOriginal.PredictLanguages(_sentenceUkr, NumLang).First().Language;
	}

	[Benchmark]
	public LinguaLanguage PanlingoDefaultFirstPart() {
		return _linguaOriginal.PredictLanguages(_sentenceUkr.Substring(0, _sentenceUkrSplitIndex), NumLang).First().Language;
	}

	[Benchmark]
	public Language PanlingoOpt() {
		return _lingua.PredictLanguages(_sentenceUkr, NumLang)[0].Language;
	}

	[Benchmark]
	public Language PanlingoOptBytes() {
		return _lingua.PredictLanguages(SentenceUkrBytes, NumLang)[0].Language;
	}

	[Benchmark]
	public Language PanlingoOptCharsFirstPart() {
		return _lingua.PredictLanguages(_sentenceUkr.AsSpan(0, _sentenceUkrSplitIndex), NumLang)[0].Language;
	}

	[Benchmark]
	public Language PanlingoOptBytesFirstPart() {
		return _lingua.PredictLanguages(SentenceUkrBytes.Slice(0, _sentenceUkrBytesSplitIndex), NumLang)[0].Language;
	}

	[Benchmark]
	public Language PanlingoOptBytesFirstPartNoCopy() {
		return _lingua.PredictLanguagesCore(new ReadOnlySpan<Byte>(_utf8Zt), NumLang)[0].Language;
	}
}