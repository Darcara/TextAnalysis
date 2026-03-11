namespace TextAnalysis.Test.Translation;

using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using IsoEnums.Iso639;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Neco.Common;
using TextAnalysis.SentenceSplitting;
using TextAnalysis.Translation;

[TestFixture]
public class MultiTranslatorTests : ATest {
	private const String _encoderDecoderMergedFile = "encdec_model.onnx";

	private static IEnumerable<TestCaseData<ModelDefinition, SessionConfiguration>> CreateLargeTestCases() {
		yield return CreateTestCase(TestData.TranslationModels.EnDe, SessionConfiguration.DefaultCpu, "CPU");

		yield return CreateTestCase(TestData.TranslationModels.EnDe, SessionConfiguration.DefaultCpu with { ExecutionProvider = ExecutionProvider.DirectML }, "DirectML");
	}

	private static IEnumerable<TestCaseData<ModelDefinition, SessionConfiguration>> CreatePerformanceTestCases() {
		// Uneven numbers like 3 usually see a performance regression
		Int32[] batchSizes = [1, 2, 4, 8];
		Int32[] interOpThreads = [1, 2, 4, 8];
		Int32[] intraOpThreads = [1, 2, 4, 8];
		var config = SessionConfiguration.DefaultCpu with { OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
		foreach (Int32 batchSize in batchSizes) {
			foreach (Int32 interOpThread in interOpThreads) {
				foreach (Int32 intraOpThread in intraOpThreads) {
					yield return CreateTestCase(TestData.TranslationModels.EnDe, config with { Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = batchSize }, IntraOpNumThreads = intraOpThread, InterOpNumThreads = intraOpThread }, $"CPU-{batchSize}Batch-{interOpThread}Inter-{intraOpThread}Intra");
				}
			}
		}

		foreach (Int32 batchSize in batchSizes) {
			yield return CreateTestCase(TestData.TranslationModels.EnDe, config with { ExecutionProvider = ExecutionProvider.DirectML, Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = batchSize } }, $"GPU-{batchSize}Batch");
		}
	}

	private static IEnumerable<TestCaseData<ModelDefinition, SessionConfiguration>> CreatePerformanceTestCasesWithPreparedModels() {
		var config = SessionConfiguration.DefaultCpu with {
			FreeDimensionOverrides = [new FreeDimensionOverride(DimensionOverrideType.ByName, "decoder_sequence_length", 1)], 
		};
		// yield return CreateTestCase(TestData.TranslationModels.EnDe, SessionConfiguration.DefaultCpu with { Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = 1 }, IntraOpNumThreads = 1, InterOpNumThreads = 1 }, "CPU-1Batch-1Inter-1Intra");
		yield return CreateTestCase(TestData.TranslationModels.EnDe, config with { Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = 1 }, IntraOpNumThreads = 1, InterOpNumThreads = 1, OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL }, "CPU-OPT-1Batch-1Inter-1Intra");
		yield return CreateTestCase(TestData.TranslationModels.EnDe, config with { Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = 1 }, IntraOpNumThreads = 1, InterOpNumThreads = 1, OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL, IntraOpThreadAffinities = "1" }, "CPU-OPT-1Batch-1Inter-1Intra-1AFF");
		yield return CreateTestCase(TestData.TranslationModels.EnDe, config with { Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = 1 }, IntraOpNumThreads = 4, InterOpNumThreads = 1, OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL }, "CPU-OPT-1Batch-1Inter-4Intra");
		yield return CreateTestCase(TestData.TranslationModels.EnDe, config with { Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = 1 }, IntraOpNumThreads = 4, InterOpNumThreads = 1, OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL, IntraOpThreadAffinities = "3;5;7" }, "CPU-OPT-1Batch-1Inter-4Intra-4AFF");
		yield return CreateTestCase(TestData.TranslationModels.EnDe, config with { Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = 4 }, IntraOpNumThreads = 1, InterOpNumThreads = 1, OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL }, "CPU-OPT-4Batch-1Inter-1Intra");
		yield return CreateTestCase(TestData.TranslationModels.EnDe, config with { Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = 4 }, IntraOpNumThreads = 1, InterOpNumThreads = 1, OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL, IntraOpThreadAffinities = "1" }, "CPU-OPT-4Batch-1Inter-1Intra-1AFF");
		yield return CreateTestCase(TestData.TranslationModels.EnDe, config with { Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = 4 }, IntraOpNumThreads = 4, InterOpNumThreads = 1, OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL }, "CPU-OPT-4Batch-1Inter-4Intra");
		yield return CreateTestCase(TestData.TranslationModels.EnDe, config with { Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = 4 }, IntraOpNumThreads = 4, InterOpNumThreads = 1, OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL, IntraOpThreadAffinities = "3;5;7" }, "CPU-OPT-4Batch-1Inter-4Intra-4AFF");

		yield return CreateTestCase(TestData.TranslationModels.EnDe with { ModelDirectoryOverride = "E:\\repos\\Experiments\\HfConvert\\en-de\\onnx-3-20" }, config with { Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = 1 }, IntraOpNumThreads = 1, InterOpNumThreads = 1, OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL }, "CPU-OPT-V20-1Batch-1Inter-1Intra");
		yield return CreateTestCase(TestData.TranslationModels.EnDe with { ModelDirectoryOverride = "E:\\repos\\Experiments\\HfConvert\\en-de\\onnx-3-20-opt" }, config with { Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = 1 }, IntraOpNumThreads = 1, InterOpNumThreads = 1, OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL }, "CPU-PREOPT-V20-1Batch-1Inter-1Intra");
		yield return CreateTestCase(TestData.TranslationModels.EnDe with { ModelDirectoryOverride = "E:\\repos\\Experiments\\HfConvert\\en-de\\onnx-3-20-dynquant" }, config with { Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = 1 }, IntraOpNumThreads = 1, InterOpNumThreads = 1, OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL }, "CPU-DynQuant-V20-1Batch-1Inter-1Intra");
		yield return CreateTestCase(TestData.TranslationModels.EnDe with { ModelDirectoryOverride = "E:\\repos\\Experiments\\HfConvert\\en-de\\onnx-3-20" }, config with { Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = 4 }, IntraOpNumThreads = 1, InterOpNumThreads = 1, OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL }, "CPU-OPT-V20-4Batch-1Inter-1Intra");
		yield return CreateTestCase(TestData.TranslationModels.EnDe with { ModelDirectoryOverride = "E:\\repos\\Experiments\\HfConvert\\en-de\\onnx-3-20" }, config with { Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = 1 }, IntraOpNumThreads = 4, InterOpNumThreads = 1, OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL }, "CPU-OPT-V20-1Batch-1Inter-4Intra");
		yield return CreateTestCase(TestData.TranslationModels.EnDe with { ModelDirectoryOverride = "E:\\repos\\Experiments\\HfConvert\\en-de\\onnx-3-20" }, config with { Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = 4 }, IntraOpNumThreads = 4, InterOpNumThreads = 1, OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL }, "CPU-OPT-V20-4Batch-1Inter-4Intra");

		// Int32[] batchSizes = [1, 2, 4, 8];
		// Int32[] interOpThreads = [1, 2, 4, 8];
		// Int32[] intraOpThreads = [1, 2, 4, 8];
		// foreach (Int32 batchSize in batchSizes) {
		// 	foreach (Int32 interOpThread in interOpThreads) {
		// 		foreach (Int32 intraOpThread in intraOpThreads) {
		// 			yield return CreateTestCase(TestData.TranslationModels.EnDe, SessionConfiguration.DefaultCpu with { Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = batchSize }, IntraOpNumThreads = intraOpThread, InterOpNumThreads = intraOpThread }, $"CPU-{batchSize}Batch-{interOpThread}Inter-{intraOpThread}Intra");
		// 		}
		// 	}
		// }
		//
		// foreach (Int32 batchSize in batchSizes) {
		// 	yield return CreateTestCase(TestData.TranslationModels.EnDe, SessionConfiguration.DefaultCpu with { ExecutionProvider = ExecutionProvider.DirectML, Batching = new BatchingConfiguration { BatchingDimensionName = "batch_size", BatchSize = batchSize } }, $"GPU-{batchSize}Batch");
		// }
	}

	private static TestCaseData<ModelDefinition, SessionConfiguration> CreateTestCase(ModelDefinition model, SessionConfiguration config, String name) {
		return new TestCaseData<ModelDefinition, SessionConfiguration>(model, config) {
			TestName = name,
		};
	}

	[SetUp]
	public void PrepareModels() {
		ModelDefinition model = TestData.TranslationModels.DeEn;
		if (!File.Exists(Path.Combine(model.ModelDirectory, _encoderDecoderMergedFile)))
			OnnxModelInspector.Merge(Path.Combine(model.ModelDirectory, model.EncoderModelName), Path.Combine(model.ModelDirectory, model.DecoderModelName), Path.Combine(model.ModelDirectory, _encoderDecoderMergedFile));
	}

	[Test]
	public void CanTranslateTwoLanguages() {
		TranslationModelLoader modelLoader = new(logger: LogFactory.CreateLogger<TranslationModelLoader>());
		using ITranslator translatorEnDe = modelLoader.Load(TestData.TranslationModels.EnDe, SessionConfiguration.DefaultCpu);
		using ITranslator translatorDeEn = modelLoader.Load(TestData.TranslationModels.DeEn with { Type = TranslationModelType.Separate2ModelsWithPast, EncoderModelNameOverride = _encoderDecoderMergedFile }, SessionConfiguration.DefaultCpu);

		SelfTestResults endeSelfTestResults = translatorEnDe.SelfTest();
		Logger.LogInformation("{SelfTestResults}", endeSelfTestResults);
		endeSelfTestResults.Success.Should().BeTrue();

		SelfTestResults deenSelfTestResults = translatorDeEn.SelfTest();
		Logger.LogInformation("{SelfTestResults}", deenSelfTestResults);
		deenSelfTestResults.Success.Should().BeTrue();

		MultiTranslatorConfiguration conf = new([translatorEnDe, translatorDeEn]);
		using MultiTranslator multi = new(conf, LogFactory.CreateLogger<MultiTranslator>());

		SelfTestResults multiResults = multi.SelfTest();
		Logger.LogInformation("{SelfTestResults}", multiResults);
		multiResults.Success.Should().BeTrue();

		String enText = "Helsinki-NLP refers to the language technology research group at the University of Helsinki. Here, we publish various resource related to multilingual NLP, machine translation, text simplification to name a few application areas. We focus on wide language coverage, open data sets and public pre-trained models.";
		String enTranslated = multi.Translate(enText, Language.English, Language.German);

		Logger.LogInformation("{Text}", enText);
		Logger.LogInformation("{Text}", enTranslated);

		String deText = "Helsinki-NLP bezieht sich auf die Forschungsgruppe Sprachtechnologie an der Universität Helsinki. Hier veröffentlichen wir verschiedene Ressourcen rund um mehrsprachiges NLP, maschinelle Übersetzung, Textvereinfachung, um einige Anwendungsbereiche zu nennen. Wir konzentrieren uns auf eine breite Sprachabdeckung, offene Datensätze und öffentliche vortrainierte Modelle.";
		String deTranslated = multi.Translate(deText, Language.German, Language.English);

		Logger.LogInformation("{Text}", deText);
		Logger.LogInformation("{Text}", deTranslated);
	}

	[TestCaseSource(nameof(CreateLargeTestCases))]
	public void CanTranslateMoreThan512Tokens(ModelDefinition endeModel, SessionConfiguration config) {
		TranslationModelLoader modelLoader = new(logger: LogFactory.CreateLogger<TranslationModelLoader>());
		using ITranslator translatorEnDe = modelLoader.Load(endeModel, config);
		MultiTranslatorConfiguration conf = new([translatorEnDe]);
		using MultiTranslator multi = new(conf, LogFactory.CreateLogger<MultiTranslator>());

		var exception = Assert.Throws<ArgumentOutOfRangeException>(() => multi.Translate(TestData.ExampleText.TomSawyerChapter2, Language.English, Language.German));
		Console.WriteLine(exception);

		ISentenceSplitter splitter = new SatSplitter(TestData.SentencePieceModels.XlmRobertaBase.Value, TestData.SentenceSplitModels.Sat3Lsm.Value, SessionConfiguration.DefaultCpu, LogFactory.CreateLogger<SatSplitter>());
		Logger.LogInformation("Starting sentence split");
		String[] sentences = splitter.Split(TestData.ExampleText.TomSawyerChapter2);
		Logger.LogInformation("Splitting done: {NumSentences} sentences", sentences.Length);
		String[] result = multi.Translate(sentences, Language.English, Language.German).ToArray();

		Logger.LogInformation(" IN: {SentencesLength}", sentences.Length);
		Logger.LogInformation("OUT: {ResultLength}", result.Length);
		for (int i = 0; i < Math.Max(sentences.Length, result.Length); i++) {
			Console.WriteLine($"{i,3} {(i >= sentences.Length ? " ? " : sentences[i].ReplaceLineEndings(" ").Trim())}");
			Console.WriteLine($"{i,3} {(i >= result.Length ? " ? " : result[i].ReplaceLineEndings(" ").Trim())}");
		}

		result.Length.Should().Be(sentences.Length);
	}

	[TestCaseSource(nameof(CreateLargeTestCases))]
	public void CanTranslateMoreThan512TokensWithUtf8(ModelDefinition endeModel, SessionConfiguration config) {
		TranslationModelLoader modelLoader = new(logger: LogFactory.CreateLogger<TranslationModelLoader>());
		using ITranslator translatorEnDe = modelLoader.Load(endeModel, config);
		MultiTranslatorConfiguration conf = new([translatorEnDe]);
		using MultiTranslator multi = new(conf, LogFactory.CreateLogger<MultiTranslator>());

		Byte[] textBytes = MagicNumbers.Utf8NoBom.GetBytes(TestData.ExampleText.TomSawyerChapter2);
		ISentenceSplitter splitter = new SatSplitter(TestData.SentencePieceModels.XlmRobertaBase.Value, TestData.SentenceSplitModels.Sat3Lsm.Value, SessionConfiguration.DefaultCpu, LogFactory.CreateLogger<SatSplitter>());
		Logger.LogInformation("Starting sentence split");
		Int32[] sentences = splitter.Split(textBytes);
		Logger.LogInformation("Splitting done: {NumSentences} sentences", sentences.Length);

		String[] result = multi.Translate(textBytes, sentences, Language.English, Language.German).ToArray();

		Logger.LogInformation(" IN: {SentencesLength}", sentences.Length);
		Logger.LogInformation("OUT: {ResultLength}", result.Length);
		for (int i = 0; i < Math.Max(sentences.Length, result.Length); i++) {
			Console.WriteLine($"{i,3} {(i >= sentences.Length ? " ? " : sentences[i])}");
			Console.WriteLine($"{i,3} {(i >= result.Length ? " ? " : result[i].ReplaceLineEndings(" ").Trim())}");
		}

		result.Length.Should().Be(sentences.Length);
	}

	[Category("Benchmark")]
	[TestCaseSource(nameof(CreatePerformanceTestCases))]
	public void LargeTextBenchmark(ModelDefinition endeModel, SessionConfiguration config) {
		TranslationModelLoader modelLoader = new(logger: LogFactory.CreateLogger<TranslationModelLoader>());
		using ITranslator translator = modelLoader.Load(endeModel, config);

		Byte[] textBytes = MagicNumbers.Utf8NoBom.GetBytes(TestData.ExampleText.TomSawyerChapter2);
		ISentenceSplitter splitter = new SatSplitter(TestData.SentencePieceModels.XlmRobertaBase.Value, TestData.SentenceSplitModels.Sat3Lsm.Value, SessionConfiguration.DefaultCpu, LogFactory.CreateLogger<SatSplitter>());
		Logger.LogInformation("Starting sentence split");
		Int32[] sentences = splitter.Split(textBytes);
		Logger.LogInformation("Splitting done: {NumSentences} sentences", sentences.Length);

		String[] result = translator.Translate(textBytes, sentences, Language.English, Language.German).ToArray();

		Logger.LogInformation(" IN: {SentencesLength}", sentences.Length);
		Logger.LogInformation("OUT: {ResultLength}", result.Length);
		for (int i = 0; i < Math.Max(sentences.Length, result.Length); i++) {
			Console.WriteLine($"{i,3} {(i >= sentences.Length ? " ? " : sentences[i])}");
			Console.WriteLine($"{i,3} {(i >= result.Length ? " ? " : result[i].ReplaceLineEndings(" ").Trim())}");
		}

		Assert.Pass();
	}

	// See also: https://learn.microsoft.com/en-us/windows/win32/procthread/quality-of-service
	[DllImport("kernel32.dll")]
	static extern int GetCurrentThreadId();

	[Category("Benchmark")]
	[TestCaseSource(nameof(CreatePerformanceTestCasesWithPreparedModels))]
	public void LargeTextBenchmarkWithPreparedModels(ModelDefinition endeModel, SessionConfiguration config) {
		if (config.IntraOpThreadAffinities != null) {
			var thisThreadId = GetCurrentThreadId();
			ProcessThread thread = Process.GetCurrentProcess().Threads.Cast<ProcessThread>().First(t => t.Id == thisThreadId);
			thread.ProcessorAffinity = 1;
		}
		TranslationModelLoader modelLoader = new(logger: LogFactory.CreateLogger<TranslationModelLoader>());
		using ITranslator translator = modelLoader.Load(endeModel, config);

		Byte[] textBytes = MagicNumbers.Utf8NoBom.GetBytes(TestData.ExampleText.TomSawyerChapter2);
		ISentenceSplitter splitter = new SatSplitter(TestData.SentencePieceModels.XlmRobertaBase.Value, TestData.SentenceSplitModels.Sat3Lsm.Value, SessionConfiguration.DefaultCpu, LogFactory.CreateLogger<SatSplitter>());
		Logger.LogInformation("Starting sentence split");
		Int32[] sentences = splitter.Split(textBytes);
		Logger.LogInformation("Splitting done: {NumSentences} sentences", sentences.Length);

		String[] result = translator.Translate(textBytes, sentences, Language.English, Language.German).ToArray();

		Logger.LogInformation(" IN: {SentencesLength}", sentences.Length);
		Logger.LogInformation("OUT: {ResultLength}", result.Length);
		for (int i = 0; i < Math.Max(sentences.Length, result.Length); i++) {
			Console.WriteLine($"{i,3} {(i >= sentences.Length ? " ? " : sentences[i])}");
			Console.WriteLine($"{i,3} {(i >= result.Length ? " ? " : result[i].ReplaceLineEndings(" ").Trim())}");
		}

		Assert.Pass();
	}
}