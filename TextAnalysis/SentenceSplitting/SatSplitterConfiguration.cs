namespace TextAnalysis.SentenceSplitting;

public sealed class SatSplitterConfiguration {
	public String XlmRobertaTokenizerModelFile { get; set; }
	public String SatModelFile { get; set; }
	public SessionConfiguration OnnxSessionConfiguration { get; set; }

	public SatSplitterConfiguration(String xlmRobertaTokenizerModelFile, String satModelFile, SessionConfiguration onnxSessionConfiguration) {
		XlmRobertaTokenizerModelFile = xlmRobertaTokenizerModelFile;
		SatModelFile = satModelFile;
		OnnxSessionConfiguration = onnxSessionConfiguration;
	}
}