namespace TextAnalysis;

using System.Numerics.Tensors;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Neco.Common;
using SentencePieceTokenizer;

public interface ISentenceSplitter : IDisposable {
	public String[] Split(String text);
	public Int32[] Split(ReadOnlySpan<Byte> utf8Bytes);
}

/// <summary>
/// Splits text in any language, using the state of the art SaT-Model - Segment any Text
/// <para>Supports 85 languages, the list is <see href="https://github.com/segment-any-text/wtpsplit#supported-languages">available on the projects github</see></para>
/// </summary>
/// <remarks>Supports all layer models, including the 'supervised mixture' (sm) variants. Does not support LoRA</remarks>
/// <seealso href="https://github.com/segment-any-text/wtpsplit">Segment any Text on Github</seealso>
public sealed class SatSplitter : ISentenceSplitter {
	private readonly ILogger<SatSplitter> _logger;
	private readonly XlmRobertaTokenizer _tokenizer;
	private readonly Boolean _disposeTokenizer;
	private readonly OnnxSession _sat;

	public SatSplitter(String xlmRobertaTokenizerModelFile, String satModelFile, SessionConfiguration satConfig, ILogger<SatSplitter>? logger) {
		_logger = logger ?? NullLogger<SatSplitter>.Instance;
		_tokenizer = new(xlmRobertaTokenizerModelFile);
		_disposeTokenizer = true;

		OnnxModelLoader ml = new(_logger);
		_sat = ml.Load(satModelFile, satConfig);
	}

	public SatSplitter(XlmRobertaTokenizer tokenizer, OnnxSession sat, ILogger<SatSplitter>? logger = null) {
		ArgumentNullException.ThrowIfNull(tokenizer);
		ArgumentNullException.ThrowIfNull(sat);

		_tokenizer = tokenizer;
		_sat = sat;
		_logger = logger ?? NullLogger<SatSplitter>.Instance;
		_disposeTokenizer = false;
	}

	#region Implementation of ISentenceSplitter

	/// <inheritdoc />
	public String[] Split(String text) {
		ReadOnlySpan<Byte> utf8Bytes = MagicNumbers.Utf8NoBom.GetBytes(text);
		Int32[] indices = Split(utf8Bytes);

		// the last sentence might not end at the end of the text 
		Int32 resultSentences = indices[^1] == utf8Bytes.Length ? indices.Length : indices.Length + 1;
		String[] sentences = new String[resultSentences];
		for (Int32 i = 0; i < indices.Length; i++) {
			Int32 lastPosition = i == 0 ? 0 : indices[i - 1];
			sentences[i] = MagicNumbers.Utf8NoBom.GetString(utf8Bytes.Slice(lastPosition, indices[i] - lastPosition));
		}

		// append the rest of the text / partial sentence
		if (resultSentences > indices.Length)
			sentences[^1] = MagicNumbers.Utf8NoBom.GetString(utf8Bytes.Slice(indices[^1]));

		return sentences;
	}

	/// <inheritdoc />
	public Int32[] Split(ReadOnlySpan<Byte> utf8Bytes) {
		(Int32[] ids, TokenSpan[] tokens) = _tokenizer.EncodeToSpans(utf8Bytes);

		List<Int32> results = new();
		Int64[] encodedIds = new Int64[ids.Length + 2];
		HugginFaceHack.ConvertToInt64AndAdd1(ids, encodedIds.AsSpan(1));
		encodedIds[0] = _tokenizer.BeginOfSentenceToken;
		encodedIds[^1] = _tokenizer.EndOfSentenceToken;
		ReadOnlySpan<Int64> inputIdSpan = new(encodedIds);

		// Even thoug the model can use up to 514 tokens, 512 seems to be faster
		const Int32 maxSequenceLength = 512;
		Int64 batchSize = _sat.Configuration.Batching.BatchSize;
		Int32 size = maxSequenceLength * (Int32)batchSize;
		// https://github.com/segment-any-text/wtpsplit/blob/main/wtpsplit/__init__.py#L767
		// sm threshold = 0.25
		// default threshold = 0.025
		// lora threshold = 0.5
		// no-limited-lookahead(but not sm) = 0.01 
		const Single threshold = 0.25f;
		// lookahead = 48 tokens, so should we split at 512-48 ?

		Int64[] ortDataInputIds = GC.AllocateArray<Int64>(size, true);
		Float16[] ortDataAttentionMask = GC.AllocateArray<Float16>(size, true);
		Span<Single> logitsDataFloat = new(new Single[size]);
		Span<Single> resultSigmoids = new(new Single[size]);
		using OrtValue inputOrtValue = OrtValue.CreateTensorValueFromMemory(ortDataInputIds, [batchSize, maxSequenceLength]);
		using OrtValue encodedAttentionOrtValue = OrtValue.CreateTensorValueFromMemory(ortDataAttentionMask, [batchSize, maxSequenceLength]);

		Dictionary<String, OrtValue> inputs = new() {
			{ "input_ids", inputOrtValue },
			{ "attention_mask", encodedAttentionOrtValue },
		};

		Int32 currentSliceOffset = 0;
		Span<Int64> inputIds = inputOrtValue.GetTensorMutableDataAsSpan<Int64>();
		Span<Float16> inputAttentionMask = encodedAttentionOrtValue.GetTensorMutableDataAsSpan<Float16>();
		inputAttentionMask.Fill(Float16.One);
		while (encodedIds.Length - currentSliceOffset > 0) {
			Int32 currentSliceLength = Math.Min(inputIdSpan.Length - currentSliceOffset, size);
			ReadOnlySpan<Int64> currentSlice = inputIdSpan.Slice(currentSliceOffset, currentSliceLength);
			currentSlice.CopyTo(inputIds);

			// Boolean isLastSlice = currentSlice.Length < maxSequenceLength;
			Boolean isLastSlice = currentSlice[^1] == _tokenizer.EndOfSentenceToken;
			// only required for last slice
			if (isLastSlice) {
				inputAttentionMask.Slice(currentSlice.Length).Fill(Float16.Zero);
			}

			currentSlice.CopyTo(ortDataInputIds);

			using IDisposableReadOnlyCollection<OrtValue>? outputs = _sat.Session.Run(_sat.RunOptions, inputs, _sat.Session.OutputNames);
			ReadOnlySpan<Float16> logitsData = outputs[0].GetTensorDataAsSpan<Float16>();

			for (Int32 index = 0; index < logitsData.Length; index++) logitsDataFloat[index] = logitsData[index].ToFloat();

			TensorPrimitives.Sigmoid(logitsDataFloat, resultSigmoids);

			// map result sigmoids
			// we added EOS at the end, so on the last slice we must ignore it
			Int32 lengthToCheck = (isLastSlice ? currentSliceLength - 1 : currentSliceLength);
			for (Int32 i = 0; i < lengthToCheck; i++) {
				if (resultSigmoids[i] > threshold) {
					// we added BOS at the beginning, so we have to shift the index -1
					Int32 originalOffset = currentSliceOffset + i - 1;
					results.Add(tokens[originalOffset].End);
				}
			}

			currentSliceOffset += currentSliceLength;
		}

		return results.ToArray();
	}

	#endregion

	#region IDisposable

	/// <inheritdoc />
	public void Dispose() {
		if (_disposeTokenizer)
			_tokenizer.Dispose();
		_sat.Dispose();
	}

	#endregion
}