namespace TextAnalysis.Test;

using IsoEnums.Iso639;

public static class TestData {
	public static class SentencePieceModels {
		public static String XlmRobertaBase => "data/xlm-roberta-base-sentencepiece.bpe.model";
		public static String EnDeMarian => "data/en-de/source.spm";
		public static String EnDeMarianVocab => "data/en-de/vocab.json";
	}
	
	public static class SentenceSplitModels {
		public const String Sat1Lsm = "data/sat1lsm.onnx";
		public const String Sat3Lsm = "data/sat3lsm.onnx";
		public const String Sat12Lsm = "data/sat12lsm.onnx";
	}
	
	public static class TranslationModels {
		public static readonly ModelDefinition EnDe = new() {
			SourceLanguage = Language.English,
			TargetLanguage = Language.German,
			Type = TranslationModelType.Separate3ModelsWithPast,
			SourceMaxTokens = 512,
			SelfTestInput = "The children became silent and thoughtful.",
			SelfTestOutput = "Die Kinder wurden still und nachdenklich.",
			ModelDirectoryOverride = "./data/en-de/",
		};
		public static readonly ModelDefinition DeEn = new() {
			SourceLanguage = Language.German,
			TargetLanguage = Language.English,
			Type = TranslationModelType.Separate3ModelsWithPast,
			SourceMaxTokens = 512,
			SelfTestInput = "Die Kinder wurden still und nachdenklich.",
			SelfTestOutput = "The children became quiet and thoughtful.",
			ModelDirectoryOverride = "./data/de-en/",
		};
	}

	public static class LanguageDetectionModels {
		public const String FastText176 = "data/lid.176.bin";
		public const String FastText218 = "data/lid.218e.bin";
	}
	
	public static class ExampleText {
		public const String OneCharacterWord = "I";
		public const String OneTokenWord = "answer";
		public const String TwoTokenWord = "thoughtful";
		public const String ThreeTokenWord = "CHAPTER";
		public const String ShortSentence = "The children became silent and thoughtful.";
		public const String Paragraph = "A frightened look in Becky's face brought Tom to his senses and he saw that he had made a blunder. Becky was not to have gone home that night! The children became silent and thoughtful. In a moment a new burst of grief from Becky showed Tom that the thing in his mind had struck hers also -- that the Sabbath morning might be half spent before Mrs. Thatcher discovered that Becky was not at Mrs. Harper's.\n\nThe children fastened their eyes upon their bit of candle and watched it melt slowly and pitilessly away; saw the half inch of wick stand alone at last; saw the feeble flame rise and fall, climb the thin column of smoke, linger at its top a moment, and then -- the horror of utter darkness reigned!";
		public const String TomSawyerFile = "data/TomSawyer.txt";
		public static String TomSawyerText => File.ReadAllText(TomSawyerFile).Substring(7499, 386044);
		public static String TomSawyerChapter1 => File.ReadAllText(TomSawyerFile).Substring(7499, 12882);
		public static String TomSawyerChapter2 => File.ReadAllText(TomSawyerFile).Substring(20381, 10483);

	}
}