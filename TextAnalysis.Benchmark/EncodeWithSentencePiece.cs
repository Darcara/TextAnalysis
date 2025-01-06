namespace TextAnalysis.Benchmark;

using Neco.Common;
using Neco.Common.Extensions;
using SentencePieceTokenizer;
using TextAnalysis.Test;

public class EncodeWithSentencePiece {
	private SentencePieceTokenizer _tokenizer = null!;
	private String _largeText = null!;
	private readonly Int64[] _target = new Int64[1.MiB()];
	private Byte[] _shortUtf8 = null!;
	private Byte[] _paragraphUtf8 = null!;
	private Byte[] _largeUtf8 = null!;

	[GlobalSetup]
	public void Setup() {
		Helper.DownloadTestData().GetResultBlocking();
		_tokenizer = new(TestData.SentencePieceModels.XlmRobertaBase);
		_largeText = TestData.ExampleText.TomSawyerText;
		_shortUtf8 = MagicNumbers.Utf8NoBom.GetBytes(TestData.ExampleText.ShortSentence);
		_paragraphUtf8 = MagicNumbers.Utf8NoBom.GetBytes(TestData.ExampleText.Paragraph);
		_largeUtf8 = MagicNumbers.Utf8NoBom.GetBytes(_largeText);
	}

	[GlobalCleanup]
	public void Cleanup() {
		_tokenizer.Dispose();
	}

	[Benchmark(OperationsPerInvoke = 8)]
	public Int32[] TokenizationSentence() {
		return _tokenizer.EncodeToIds(TestData.ExampleText.ShortSentence);
	}

	[Benchmark(OperationsPerInvoke = 186)]
	public Int32[] TokenizationSmallParagraph() {
		return _tokenizer.EncodeToIds(TestData.ExampleText.Paragraph);
	}

	[Benchmark(OperationsPerInvoke = 105899)]
	public Int32[] TokenizationLargeText() {
		return _tokenizer.EncodeToIds(_largeText);
	}

	[Benchmark(OperationsPerInvoke = 8)]
	public (Int32[], TokenSpan[]) SpanTokenizationSentence() {
		return _tokenizer.EncodeToSpans(_shortUtf8);
	}

	[Benchmark(OperationsPerInvoke = 186)]
	public (Int32[], TokenSpan[]) SpanTokenizationSmallParagraph() {
		return _tokenizer.EncodeToSpans(_paragraphUtf8);
	}

	[Benchmark(OperationsPerInvoke = 105899)]
	public (Int32[], TokenSpan[]) SpanTokenizationLargeText() {
		return _tokenizer.EncodeToSpans(_largeUtf8);
	}

	[Benchmark(OperationsPerInvoke = 8)]
	public Int32[] TokenizationAndFixSentence() {
		Int32[] tokens = _tokenizer.EncodeToIds(TestData.ExampleText.ShortSentence);
		HugginFaceHack.ConvertToInt64AndAdd1(tokens, _target);
		return tokens;
	}

	[Benchmark(OperationsPerInvoke = 186)]
	public Int32[] TokenizationAndFixSmallParagraph() {
		Int32[] tokens = _tokenizer.EncodeToIds(TestData.ExampleText.Paragraph);
		HugginFaceHack.ConvertToInt64AndAdd1(tokens, _target);
		return tokens;
	}

	[Benchmark(OperationsPerInvoke = 105899)]
	public Int32[] TokenizationAndFixLargeText() {
		Int32[] tokens = _tokenizer.EncodeToIds(_largeText);
		HugginFaceHack.ConvertToInt64AndAdd1(tokens, _target);
		return tokens;
	}
}