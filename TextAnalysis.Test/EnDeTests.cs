namespace TextAnalysis.Test;

using FluentAssertions;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;

public class EnDeTests : ATest {
	private ModelDefinition _endeDefinition;

	[SetUp]
	public void Setup() {
		_endeDefinition = new ModelDefinition {
			SourceLanguage = "en",
			TargetLanguage = "de",
			SourceMaxTokens = 512,
			SelfTestInput = "The children became silent and thoughtful.",
			SelfTestOutput = "Die Kinder wurden still und nachdenklich.",
			ModelDirectoryOverride = "./data/en-de/",
		};
	}

	[Test]
	public void LoadsBaseCorrectly() {
		TranslationModelLoader modelLoader = new(LogFactory.CreateLogger<TranslationModelLoader>());
		using TranslationModel translator = modelLoader.Load(_endeDefinition, SessionConfiguration.DefaultCpu);
		translator.Should().NotBeNull();
		SelfTestResults selfTestResults = translator.SelfTest();
		Logger.LogInformation("{SelfTestResults}", selfTestResults);
		selfTestResults.Success.Should().BeTrue();
	}

	[Test]
	public void LoadsOptimizedCorrectly() {
		TranslationModelLoader modelLoader = new(LogFactory.CreateLogger<TranslationModelLoader>());
		using TranslationModel translator = modelLoader.Load(_endeDefinition, SessionConfiguration.DefaultCpu with { OptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL });
		translator.Should().NotBeNull();
		SelfTestResults selfTestResults = translator.SelfTest();
		Logger.LogInformation("{SelfTestResults}", selfTestResults);
		selfTestResults.Success.Should().BeTrue();
	}
}