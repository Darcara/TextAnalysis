namespace TextAnalysis.Test;

[SetUpFixture]
public class GlobalTestSetup {
	[OneTimeSetUp]
	public void Setup() {
		// Just to get the FluentAssertion / Xceed Licence text out of the test reports
		TextWriter textWriter = Console.Out;
		Console.SetOut(TextWriter.Null);
		String.Empty.Should().BeNullOrWhiteSpace();
		Console.SetOut(textWriter);
	}
}