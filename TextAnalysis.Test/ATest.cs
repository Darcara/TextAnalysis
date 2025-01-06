namespace TextAnalysis.Test;

using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;
using Neco.Common.Extensions;

[TestFixture]
public abstract class ATest {
	
	public ILoggerFactory LogFactory;

	public ILogger Logger => LogFactory.CreateLogger("Test");
	
	[SetUp]
	public void CreateLogger() {
		LogFactory = LoggerFactory.Create(builder => builder
			.SetMinimumLevel(LogLevel.Debug)
			.AddSimpleConsole(conf => {
				conf.ColorBehavior = LoggerColorBehavior.Enabled;
				conf.SingleLine = false;
			}));
	}

	[TearDown]
	public void DisposeFactory() {
		LogFactory.Dispose();
	}
	
	[SetUp]
	public void EnsureTestModelAvailable() => Helper.DownloadTestData().GetResultBlocking();

	[SetUp]
	public void EnsureNativeFilesPresent() => Helper.EnsureNativeFilesPresent();

	

}