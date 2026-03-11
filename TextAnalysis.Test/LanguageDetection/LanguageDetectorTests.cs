namespace TextAnalysis.Test.LanguageDetection;

using IsoEnums.Iso639;
using Neco.Common;
using TextAnalysis.LanguageDetection;

[TestFixtureSource(typeof(LanguageDetectorTests), nameof(LanguageDetectors))]
internal class LanguageDetectorTests : ATest, IDisposable {
	private readonly ILanguageDetector _detector;
	// Location of '?' in utf8 = 18 - 51 - 82
	private static ReadOnlySpan<Byte> Sentences => "Hello, how are you? Привіт, як справи? Привет, как дела?"u8;

	public static IEnumerable<TestFixtureData> LanguageDetectors {
		get {
		yield return new TestFixtureData {Detector=new LinguaLanguageDetector(), Name = "Lingua-High"};
		yield return new TestFixtureData {Detector=new LinguaLanguageDetector(true), Name = "Lingua-Low"};
		yield return new TestFixtureData {Detector=new FastTextLanguageDetector(null), Name = "FastText-Low"};
		yield return new TestFixtureData {Detector=new FastTextLanguageDetector(TestData.LanguageDetectionModels.FastText176.Value), Name = "FastText-176"};
		}
	}

	public LanguageDetectorTests(TestFixtureData detector) {
		_detector = detector.Detector;
	}
	
	[TestCase("Hello, how are you?", Language.English)]
	[TestCase("Привіт, як справи?", Language.Ukrainian)]
	[TestCase("Привет, как дела?", Language.Russian)]
	public void CanDetectFromString(String text, Language language) {
		LanguagePrediction bestPrediction = _detector.Detect(text);
		LanguagePrediction bestPrediction2 = _detector.Detect(text.AsSpan());
		LanguagePrediction[] predictions = _detector.Detect(text, 10);
		LanguagePrediction[] predictions3 = _detector.Detect(text,3);
		Console.WriteLine(text);
		Console.WriteLine(String.Join(Environment.NewLine, predictions));
		predictions.First().Language.Should().Be(language);
		predictions3.First().Language.Should().Be(language);
		bestPrediction.Language.Should().Be(bestPrediction2.Language);
		bestPrediction.Confidence.Should().BeApproximately(bestPrediction2.Confidence, 0.0001);
		bestPrediction.Language.Should().Be(predictions.First().Language);
		bestPrediction.Confidence.Should().BeApproximately(predictions.First().Confidence, 0.0001);
		bestPrediction.Language.Should().Be(predictions3.First().Language);
		bestPrediction.Confidence.Should().BeApproximately(predictions3.First().Confidence, 0.0001);
		
		Console.WriteLine("----");
		Console.WriteLine(text.Substring(0,6));
		predictions = _detector.Detect(text.AsSpan(0,6), 10);
		Console.WriteLine(String.Join(Environment.NewLine, predictions));
	}
	
	[TestCase(0,19, Language.English)]
	[TestCase(20,32, Language.Ukrainian)]
	[TestCase(53,30, Language.Russian)]
	public void CanDetectFromBytes(Int32 start, Int32 length, Language language) {
		LanguagePrediction bestPrediction = _detector.Detect(Sentences.Slice(start, length));
		LanguagePrediction[] predictions = _detector.Detect(Sentences.Slice(start, length), 10);
		Console.WriteLine(MagicNumbers.Utf8NoBom.GetString(Sentences.Slice(start, length)));
		Console.WriteLine(String.Join(Environment.NewLine, predictions));
		predictions.First().Language.Should().Be(language);
		bestPrediction.Language.Should().Be(predictions.First().Language);
		bestPrediction.Confidence.Should().BeApproximately(predictions.First().Confidence, 0.0001);
		
		Console.WriteLine("----");
		Console.WriteLine(MagicNumbers.Utf8NoBom.GetString(Sentences.Slice(start, 12)));
		predictions = _detector.Detect(Sentences.Slice(start,12), 10);
		Console.WriteLine(String.Join(Environment.NewLine, predictions));
	}

	#region IDisposable

	/// <inheritdoc />
	public void Dispose() {
		Console.WriteLine("Disposing detector");
		_detector.Dispose();
	}

	#endregion
}

internal sealed class TestFixtureData {
	public required ILanguageDetector Detector;
	public required String Name;

	#region Overrides of Object

	/// <inheritdoc />
	public override String ToString() => Name;

	#endregion
}