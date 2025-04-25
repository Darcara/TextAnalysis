namespace TextAnalysis.Test.LanguageDetection;

using IsoEnums.Iso639;
using Neco.Common;
using TextAnalysis.LanguageDetection;

[TestFixture]
public class CustomLinguaDetectorTests {
	// Location of '?' in utf8 = 18 - 51 - 82
	private static ReadOnlySpan<Byte> Sentences => "Hello, how are you? Привіт, як справи? Привет, как дела?"u8;
	
	[TestCase("Hello, how are you?", Language.English)]
	[TestCase("Привіт, як справи?", Language.Ukrainian)]
	[TestCase("Привет, как дела?", Language.Russian)]
	public void CanDetectFromString(String text, Language language) {
		using var detector = new CustomLinguaDetector();
		LanguagePrediction[] predictions = detector.PredictLanguages(text);
		LanguagePrediction[] predictions3 = detector.PredictLanguages(text,3);
		Console.WriteLine(text);
		Console.WriteLine(String.Join(Environment.NewLine, predictions));
		predictions.First().Language.Should().Be(language);
		predictions3.First().Language.Should().Be(language);
		
		Console.WriteLine("----");
		Console.WriteLine(text.Substring(0,6));
		predictions = detector.PredictLanguages(text.AsSpan(0,6));
		Console.WriteLine(String.Join(Environment.NewLine, predictions));
	}
	
	[TestCase(0,19, Language.English)]
	[TestCase(20,32, Language.Ukrainian)]
	[TestCase(53,30, Language.Russian)]
	public void CanDetectFromBytes(Int32 start, Int32 length, Language language) {
		using var detector = new CustomLinguaDetector();
		LanguagePrediction[] predictions = detector.PredictLanguages(Sentences.Slice(start, length));
		Console.WriteLine(MagicNumbers.Utf8NoBom.GetString(Sentences.Slice(start, length)));
		Console.WriteLine(String.Join(Environment.NewLine, predictions));
		predictions.First().Language.Should().Be(language);
		
		Console.WriteLine("----");
		Console.WriteLine(MagicNumbers.Utf8NoBom.GetString(Sentences.Slice(start, 12)));
		predictions = detector.PredictLanguages(Sentences.Slice(start,12));
		Console.WriteLine(String.Join(Environment.NewLine, predictions));
	}
}