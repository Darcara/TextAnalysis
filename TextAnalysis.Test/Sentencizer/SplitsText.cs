namespace TextAnalysis.Test.Sentencizer;

using System.Diagnostics;
using System.Numerics.Tensors;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using SentencePieceTokenizer;

[TestFixture]
public class SplitsText : ATest {
	[TestCase("This is a sentence. This is another sentence.")]
	[TestCase("This is a test This is another test.")]
	[TestCase("Hello this is a test But this is different now Now the next one starts looool")]
	[TestCase("In linguistics and grammar, a sentence is a linguistic expression, such as the English example \\\"The quick brown fox jumps over the lazy dog.\\\" In traditional grammar, it is typically defined as a string of words that expresses a complete thought, or as a unit consisting of a subject and predicate. In non-functional linguistics it is typically defined as a maximal unit of syntactic structure such as a constituent. In functional linguistics, it is defined as a unit of written texts delimited by graphological features such as upper-case letters and markers such as periods, question marks, and exclamation marks. This notion contrasts with a curve, which is delimited by phonologic features such as pitch and loudness and markers such as pauses; and with a clause, which is a sequence of words that represents some process going on throughout time.[1] A sentence can include words grouped meaningfully to express a statement, question, exclamation, request, command, or suggestion.[2]", TestName = "WikiText")]
	[TestCase(TestData.ExampleText.OneCharacterWord)]
	[TestCase(TestData.ExampleText.OneTokenWord)]
	[TestCase(TestData.ExampleText.TwoTokenWord)]
	[TestCase(TestData.ExampleText.ThreeTokenWord)]
	[TestCase(TestData.ExampleText.ShortSentence, TestName = nameof(TestData.ExampleText.ShortSentence))]
	[TestCase(TestData.ExampleText.Paragraph, TestName = nameof(TestData.ExampleText.Paragraph))]
	public void SplitsEnglishText(String text) {
		// paper: https://arxiv.org/abs/2406.16678
		// SaT Base Models: Base SaT (Segment any Text) models, to be used for sentence and paragraph segmentation. Easily adaptable via LoRA.
		// SaT Supervised Mixture (SM) Models: SaT (Segment any Text) models, further trained on a Supervised Mixture of diverse styles and corruptions. Universal Sentence Segmentation models!

		using XlmRobertaTokenizer tokenizer = new(TestData.SentencePieceModels.XlmRobertaBase);
		// no idea why +1, but it is needed.
		// python script prepends ClsToken and appends SepToken --> not required, but increases accuracy.
		Int32[] tokenized = tokenizer.EncodeToIds(text);
		Int64[] encodedIds = new Int64[tokenized.Length + 2];
		HugginFaceHack.ConvertToInt64AndAdd1(tokenized, encodedIds.AsSpan(1));
		encodedIds[0] = tokenizer.BeginOfSentenceToken;
		encodedIds[^1] = tokenizer.EndOfSentenceToken;

		String[] encodedStrings = ["<s>", ..tokenizer.EncodeToStrings(text), "</s>"];
		Console.WriteLine($"#{encodedIds.Length}: " + String.Join(", ", encodedIds));
		Console.WriteLine($"#{encodedStrings.Length}: " + String.Join(", ", encodedStrings));
		encodedIds.Length.Should().BeGreaterThanOrEqualTo(text.Split().Length);

		Float16[] encodedAttention = new Float16[encodedIds.Length];
		Array.Fill(encodedAttention, Float16.One);

		using InferenceSession session = new(TestData.OnnxModels.Sat3Lsm);
		using OrtValue? inputOrtValue = OrtValue.CreateTensorValueFromMemory(encodedIds, [1, encodedIds.Length]);
		using OrtValue? encodedAttentionOrtValue = OrtValue.CreateTensorValueFromMemory(encodedAttention, [1, encodedIds.Length]);

		Dictionary<String, OrtValue> inputs = new() {
			{ "input_ids", inputOrtValue },
			{ "attention_mask", encodedAttentionOrtValue },
		};

		using RunOptions runOptions = new();
		using IDisposableReadOnlyCollection<OrtValue>? outputs = session.Run(runOptions, inputs, session.OutputNames);
		ReadOnlySpan<Float16> logitsData = outputs[0].GetTensorDataAsSpan<Float16>();
		Single[] logitsDataFloat = new Single[logitsData.Length];
		for (Int32 index = 0; index < logitsData.Length; index++) logitsDataFloat[index] = logitsData[index].ToFloat();

		logitsData.Length.Should().Be(encodedIds.Length);


		Single[] sigmoids = new Single[logitsData.Length];
		TensorPrimitives.Sigmoid(logitsDataFloat, sigmoids);

		Int32 maxStringLength = encodedStrings.Max(s => s.Length);
		Console.WriteLine("*** Output ***");
		Console.WriteLine($"{"Token".PadLeft(maxStringLength)} - {"Id",7} - {"RawValue",8} - {"Sigmoid",8}");
		for (Int32 index = 0; index < encodedStrings.Length; index++) {
			String encodedString = encodedStrings[index];
			Int64 encodedId = encodedIds[index];
			Single eosProbability = logitsData[index].ToFloat();
			Console.WriteLine($"{encodedString.PadLeft(maxStringLength)} - {encodedId,7} - {eosProbability,8:N3} - {sigmoids[index],8:N3} {(sigmoids[index] > 0.25 ? "<---" : "")}");
		}
	}

	[TestCase("This is a sentence. This is another sentence.")] // Interestingly this will be split correctly by the 1-layer and 12-layer model, but not the 3-layer model.
	[TestCase("This is a test This is another test.")]
	[TestCase("Hello this is a test But this is different now Now the next one starts looool")]
	[TestCase("In linguistics and grammar, a sentence is a linguistic expression, such as the English example \\\"The quick brown fox jumps over the lazy dog.\\\" In traditional grammar, it is typically defined as a string of words that expresses a complete thought, or as a unit consisting of a subject and predicate. In non-functional linguistics it is typically defined as a maximal unit of syntactic structure such as a constituent. In functional linguistics, it is defined as a unit of written texts delimited by graphological features such as upper-case letters and markers such as periods, question marks, and exclamation marks. This notion contrasts with a curve, which is delimited by phonologic features such as pitch and loudness and markers such as pauses; and with a clause, which is a sequence of words that represents some process going on throughout time.[1] A sentence can include words grouped meaningfully to express a statement, question, exclamation, request, command, or suggestion.[2]", TestName = "WikiText")]
	[TestCase(TestData.ExampleText.OneCharacterWord)]
	[TestCase(TestData.ExampleText.OneTokenWord)]
	[TestCase(TestData.ExampleText.TwoTokenWord)]
	[TestCase(TestData.ExampleText.ThreeTokenWord)]
	[TestCase(TestData.ExampleText.ShortSentence, TestName = nameof(TestData.ExampleText.ShortSentence))]
	[TestCase(TestData.ExampleText.Paragraph, TestName = nameof(TestData.ExampleText.Paragraph))]
	public void SatSplitterExample(String text) {
		using SatSplitter splitter = new(TestData.SentencePieceModels.XlmRobertaBase, TestData.OnnxModels.Sat3Lsm, SessionConfiguration.DefaultCpu, LogFactory.CreateLogger<SatSplitter>());
		String[] sentences = splitter.Split(text);
		Console.WriteLine(text);
		Console.WriteLine();
		for (Int32 index = 0; index < sentences.Length; index++) {
			String se = sentences[index];
			Console.WriteLine($"#{index,2}: " + se.Trim());
		}
	}

	[Category("Benchmark")]
	[TestCase(TestData.OnnxModels.Sat1Lsm, 4931)]
	[TestCase(TestData.OnnxModels.Sat3Lsm, 4950)]
	[TestCase(TestData.OnnxModels.Sat12Lsm, 4992)]
	public void SplitsEnglishTextWhenSplit(String model, Int32 expectedSentences) {
		String text = TestData.ExampleText.TomSawyerText;
		SessionConfiguration config = SessionConfiguration.DefaultCpu with {
			Batching = new BatchingConfiguration { BatchSize = 4, BatchingDimensionName = "batch" },
			FreeDimensionOverrides = [new(DimensionOverrideType.ByName, "sequence", 512)],
			OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
			InterOpNumThreads = 2,
			IntraOpNumThreads = 4,
			// ExecutionProvider = ExecutionProvider.DirectML,
		};
		using SatSplitter splitter = new(TestData.SentencePieceModels.XlmRobertaBase, model, config, LogFactory.CreateLogger<SatSplitter>());
		Stopwatch sw = Stopwatch.StartNew();
		String[] sentences = splitter.Split(text);
		sw.Stop();
		Console.WriteLine($"Sentences: {sentences.Length} in {sw.Elapsed.TotalSeconds}s");
		Console.WriteLine(text);
		Console.WriteLine();
		for (Int32 index = 0; index < sentences.Length; index++) {
			String se = sentences[index];
			Console.WriteLine($"#{index,2}: " + se.Trim());
		}

		sentences.Length.Should().BeCloseTo(expectedSentences, 50);
	}

	[Category("Benchmark")]
	[TestCase(1, 1, 1, false)]
	[TestCase(1, 1, 2, false)]
	[TestCase(1, 1, 4, false)]
	[TestCase(4, 1, 1, false)]
	[TestCase(4, 1, 2, false)]
	[TestCase(4, 1, 4, false)]
	[TestCase(4, 2, 2, false)]
	[TestCase(4, 2, 4, false)]
	[TestCase(4, 4, 4, false)]
	[TestCase(4, 1, 8, false)]
	[TestCase(1, 1, 1, true)]
	[TestCase(1, 1, 4, true)]
	[TestCase(2, 1, 1, true)]
	[TestCase(4, 1, 1, true)]
	[TestCase(4, 1, 4, true)]
	[TestCase(4, 4, 1, true)]
	[TestCase(4, 4, 4, true)]
	[TestCase(6, 1, 1, true)]
	[TestCase(8, 1, 1, true)]
	[TestCase(16, 1, 1, true)]
	public void Sat1LTimeEstimation(Int32 batchSize, Int32 interOpThreads, Int32 intraOpThreads, Boolean dml) {
		EstimateModel(batchSize, interOpThreads, intraOpThreads, dml, TestData.OnnxModels.Sat1Lsm);
	}

	[Category("Benchmark")]
	[TestCase(1, 1, 1, false)]
	[TestCase(1, 1, 2, false)]
	[TestCase(1, 1, 4, false)]
	[TestCase(4, 1, 1, false)]
	[TestCase(4, 1, 2, false)]
	[TestCase(4, 1, 4, false)]
	[TestCase(4, 2, 2, false)]
	[TestCase(4, 2, 4, false)]
	[TestCase(4, 4, 4, false)]
	[TestCase(4, 1, 8, false)]
	[TestCase(1, 1, 1, true)]
	[TestCase(1, 1, 4, true)]
	[TestCase(2, 1, 1, true)]
	[TestCase(4, 1, 1, true)]
	[TestCase(4, 1, 4, true)]
	[TestCase(4, 4, 1, true)]
	[TestCase(4, 4, 4, true)]
	[TestCase(6, 1, 1, true)]
	[TestCase(8, 1, 1, true)]
	[TestCase(16, 1, 1, true)]
	public void Sat3LTimeEstimation(Int32 batchSize, Int32 interOpThreads, Int32 intraOpThreads, Boolean dml) {
		EstimateModel(batchSize, interOpThreads, intraOpThreads, dml, TestData.OnnxModels.Sat3Lsm);
	}

	private void EstimateModel(Int32 batchSize, Int32 interOpThreads, Int32 intraOpThreads, Boolean dml, String model) {
		String text = TestData.ExampleText.TomSawyerText;
		SessionConfiguration config = SessionConfiguration.DefaultCpu with {
			Batching = new BatchingConfiguration { BatchSize = batchSize, BatchingDimensionName = "batch" },
			FreeDimensionOverrides = [new(DimensionOverrideType.ByName, "sequence", 512)],
			OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
			InterOpNumThreads = interOpThreads,
			IntraOpNumThreads = intraOpThreads,
			ExecutionProvider = dml ? ExecutionProvider.DirectML : ExecutionProvider.CPU,
		};
		using SatSplitter splitter = new(TestData.SentencePieceModels.XlmRobertaBase, model, config, LogFactory.CreateLogger<SatSplitter>());
		Stopwatch sw = Stopwatch.StartNew();
		String[] sentences = splitter.Split(text);
		sw.Stop();
		Console.WriteLine($"Sentences: {sentences.Length} in {sw.Elapsed.TotalSeconds}s for batch={batchSize}, inter={interOpThreads}, intra={intraOpThreads}, dml={dml}");
	}
}