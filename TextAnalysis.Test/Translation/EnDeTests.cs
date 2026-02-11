namespace TextAnalysis.Test.Translation;

using FluentAssertions;
using global::IsoEnums.Iso639;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using TextAnalysis.Translation;

internal static class TranslationTestSources {
	internal static IEnumerable<ValueTuple<ModelDefinition, String>> BaseModelsToTest() {
		yield return (TestData.TranslationModels.DeEn, "HFBase-DeEn");
		yield return (TestData.TranslationModels.DeEn with  {
			ModelDirectoryOverride = "E:\\repos\\Experiments\\HfConvert\\de-en\\onnx-3-20",
		}, "V20-DeEn");
		
		yield return (TestData.TranslationModels.EnDe, "HFBase");
		yield return (TestData.TranslationModels.EnDe with  {
			ModelDirectoryOverride = "E:\\repos\\Experiments\\HfConvert\\en-de\\onnx-3",
		}, "V14");

		yield return (TestData.TranslationModels.EnDe with  {
			ModelDirectoryOverride = "E:\\repos\\Experiments\\HfConvert\\en-de\\onnx-3-20",
		}, "V20");
	}

	public static IEnumerable<TestCaseData<ModelDefinition, Boolean, GraphOptimizationLevel>> TranslationTestCases(GraphOptimizationLevel optimizationLevel) {
		foreach (var baseModel in BaseModelsToTest()) {
			foreach (Boolean externalWeights in new[] { false, true }) {
				ModelDefinition model = externalWeights ? baseModel.Item1.WithExternalWeights() : baseModel.Item1;
				String testName = $"{baseModel.Item2}-{model.Type}x{optimizationLevel}{(externalWeights ? "-extWeights" : String.Empty)}";
				yield return new TestCaseData<ModelDefinition, Boolean, GraphOptimizationLevel>(model, externalWeights, optimizationLevel) { TestName = testName };

				model = baseModel.Item1 with {
					Type = TranslationModelType.Separate2ModelsWithPast,
					EncoderModelNameOverride = "encdec_model.onnx",
				};

				model = externalWeights ? model.WithExternalWeights() : model;
				testName = $"{baseModel.Item2}-{model.Type}x{optimizationLevel}{(externalWeights ? "-extWeights" : String.Empty)}";
				yield return new TestCaseData<ModelDefinition, Boolean, GraphOptimizationLevel>(model, externalWeights, optimizationLevel) { TestName = testName };
			}
		}
	}
}

public class EnDeTests : ATest {
	[SetUp]
	public void PrepareModels() {
		foreach (var baseModel in TranslationTestSources.BaseModelsToTest()) {
			ModelDefinition model = baseModel.Item1;
			String encoderDecoderMergedFile = "encdec_model.onnx";
			String encoderDecoderMergedFileWithExternalInitializers = "encdec_model.clear.onnx";
			String encoderWithExternalInitializers = Path.ChangeExtension(model.EncoderModelName, ".clear" + Path.GetExtension(model.EncoderModelName));
			String decoderWithExternalInitializers = Path.ChangeExtension(model.DecoderModelName, ".clear" + Path.GetExtension(model.DecoderModelName));
			String decoderWithPastWithExternalInitializers = Path.ChangeExtension(model.DecoderWithPastModelName, ".clear" + Path.GetExtension(model.DecoderModelName));

			if (!File.Exists(Path.Combine(model.ModelDirectory, encoderDecoderMergedFile)))
				OnnxModelInspector.Merge(Path.Combine(model.ModelDirectory, model.EncoderModelName), Path.Combine(model.ModelDirectory, model.DecoderModelName), Path.Combine(model.ModelDirectory, encoderDecoderMergedFile));

			if (!File.Exists(Path.Combine(model.ModelDirectory, encoderWithExternalInitializers)))
				OnnxModelInspector.Split(Path.Combine(model.ModelDirectory, model.EncoderModelName));
			if (!File.Exists(Path.Combine(model.ModelDirectory, decoderWithExternalInitializers)))
				OnnxModelInspector.Split(Path.Combine(model.ModelDirectory, model.DecoderModelName));
			if (!File.Exists(Path.Combine(model.ModelDirectory, decoderWithPastWithExternalInitializers)))
				OnnxModelInspector.Split(Path.Combine(model.ModelDirectory, model.DecoderWithPastModelName));
			if (!File.Exists(Path.Combine(model.ModelDirectory, encoderDecoderMergedFileWithExternalInitializers)))
				OnnxModelInspector.Split(Path.Combine(model.ModelDirectory, encoderDecoderMergedFile));
		}
	}

	[TestCaseSource(typeof(TranslationTestSources), nameof(TranslationTestSources.TranslationTestCases), new Object?[] { GraphOptimizationLevel.ORT_DISABLE_ALL })]
	[TestCaseSource(typeof(TranslationTestSources), nameof(TranslationTestSources.TranslationTestCases), new Object?[] { GraphOptimizationLevel.ORT_ENABLE_ALL })]
	public void LoadsCorrectly(ModelDefinition modelDefinition, Boolean externalWeights, GraphOptimizationLevel optimizationLevel) {
		if (!File.Exists(Path.Combine(modelDefinition.ModelDirectory, modelDefinition.EncoderModelName)))
			Assert.Inconclusive("Encoder missing");

		TranslationModelLoader modelLoader = new(logger: LogFactory.CreateLogger<TranslationModelLoader>());
		using ITranslator translator = modelLoader.Load(modelDefinition, SessionConfiguration.DefaultCpu with { OptimizationLevel = optimizationLevel });
		translator.Should().NotBeNull();
		SelfTestResults selfTestResults = translator.SelfTest();
		Logger.LogInformation("{SelfTestResults}", selfTestResults);
		selfTestResults.Success.Should().BeTrue();
	}
	
	[TestCaseSource(typeof(TranslationTestSources), nameof(TranslationTestSources.TranslationTestCases), new Object?[] { GraphOptimizationLevel.ORT_ENABLE_ALL })]
	public void LoadsCorrectlyDml(ModelDefinition modelDefinition, Boolean externalWeights, GraphOptimizationLevel optimizationLevel) {
		if (!File.Exists(Path.Combine(modelDefinition.ModelDirectory, modelDefinition.EncoderModelName)))
			Assert.Inconclusive("Encoder missing");
		
		TranslationModelLoader modelLoader = new(logger:LogFactory.CreateLogger<TranslationModelLoader>());
		using ITranslator translator = modelLoader.Load(modelDefinition, SessionConfiguration.DefaultCpu with { ExecutionProvider = ExecutionProvider.DirectML, EnableVerboseOrtLogging = true});
		translator.Should().NotBeNull();
		SelfTestResults selfTestResults = translator.SelfTest();
		Logger.LogInformation("{SelfTestResults}", selfTestResults);
		selfTestResults.Success.Should().BeTrue();
	}

	[TestCase("CHAPTER I")]
	[TestCase("CHAPTER II")]
	[TestCase("CHAPTER III")]
	[TestCase("CHAPTER II Saturday morning was come, and all the summer world was bright and fresh, and brimming with life. There was a song in every heart; and if the heart was young the music issued at the lips. There was cheer in every face and a spring in every step. The locust-trees were in bloom and the fragrance of the blossoms filled the air. Cardiff Hill, beyond the village and above it, was green with vegetation and it lay just far enough away to seem a Delectable Land, dreamy, reposeful, and inviting.", TestName = "TomSawyerChapter2")]
	public void ExampleTranslations(String textEn) {
		TranslationModelLoader modelLoader = new(logger: LogFactory.CreateLogger<TranslationModelLoader>());
		using ITranslator translator = modelLoader.Load(TestData.TranslationModels.EnDe, SessionConfiguration.DefaultCpu);

		String textDe = translator.Translate(textEn, Language.English, Language.German);
		Console.WriteLine(textEn);
		Console.WriteLine(textDe);

		textDe.Should().NotBeNullOrWhiteSpace();
	}
}