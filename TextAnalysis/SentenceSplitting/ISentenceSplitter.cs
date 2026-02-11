namespace TextAnalysis;

public interface ISentenceSplitter : IDisposable {
	public String[] Split(ReadOnlySpan<Char> text);
	public Int32[] SplitIndices(ReadOnlySpan<Char> text);
	public Int32[] Split(ReadOnlySpan<Byte> utf8Bytes);
}