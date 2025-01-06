namespace TextAnalysis;

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Numerics.Tensors;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using SentencePieceTokenizer;

public sealed class SelfTestResults {
	[MemberNotNullWhen(true, nameof(Input), nameof(Output))]
	public required Boolean Success { get; init; }

	public required String? Input { get; init; }
	public required String? Output { get; init; }
	public required StringBuilder Log { get; init; }

	public static SelfTestResults Fail(String message) {
		return new SelfTestResults() {
			Success = false,
			Input = null,
			Output = null,
			Log = new StringBuilder(message),
		};
	}

	#region Overrides of Object

	/// <inheritdoc />
	public override String ToString() => Log.ToString();

	#endregion
}

[DebuggerDisplay("{_modelDefinition.SourceLanguage}-{_modelDefinition.TargetLanguage}")]
public class TranslationModel : IDisposable {
	private readonly ModelDefinition _modelDefinition;
	private readonly OnnxSession _encoderSession;
	private readonly OnnxSession _decoderSession;
	private readonly OnnxSession _decoderWithPastSession;
	private readonly MarianTokenizer _tokenizer;

	public Boolean CanBeUsedConcurrently => _encoderSession.Configuration.ExecutionProvider == ExecutionProvider.CPU;

	public TranslationModel(ModelDefinition modelDefinition, OnnxSession encoderSession, OnnxSession decoderSession, OnnxSession decoderWithPastSession, MarianTokenizer tokenizer) {
		_modelDefinition = modelDefinition;
		_encoderSession = encoderSession;
		_decoderSession = decoderSession;
		_decoderWithPastSession = decoderWithPastSession;
		_tokenizer = tokenizer;
	}

	public SelfTestResults SelfTest() {
		if (String.IsNullOrEmpty(_modelDefinition.SelfTestInput)) return SelfTestResults.Fail("No Input");
		if (String.IsNullOrEmpty(_modelDefinition.SelfTestOutput)) return SelfTestResults.Fail("No Output");

		StringBuilder log = new();
		// only required for logging
		String[] encoded = _tokenizer.EncodeToStrings(_modelDefinition.SelfTestInput);
		String[] expectedEncoded = _tokenizer.EncodeToStrings(_modelDefinition.SelfTestOutput);

		Int64[] encodedIds = _tokenizer.EncodeToIds(_modelDefinition.SelfTestInput, suffixIds:[_tokenizer.EndOfSentenceToken, _tokenizer.PadToken]);
		Int64[] expectedIds = _tokenizer.EncodeToIds(_modelDefinition.SelfTestOutput);

		Int64[] encodedAttention = new Int64[encodedIds.Length];
		Array.Fill(encodedAttention, 1);
		encodedAttention[^1] = 0;

		Int32 padLength = encoded.Max(token => token.Length);
		log.AppendLine($"Testing 3-part {_modelDefinition.EncoderModelName} and {_modelDefinition.DecoderModelName} / {_modelDefinition.DecoderWithPasthModelName} with {_encoderSession.Configuration.OptimizationLevel}");
		log.AppendLine($"    Index: {String.Join(", ", Enumerable.Range(0, Math.Max(encodedIds.Length, expectedIds.Length)).Select(i => i.ToString().PadLeft(padLength)))}");
		log.AppendLine($"    Token: {String.Join(", ", encoded.Select(token => token.PadLeft(padLength)))}");
		log.AppendLine($"  Encoded: {String.Join(", ", encodedIds.Select(i => i.ToString().PadLeft(padLength)))}");
		log.AppendLine($"Attention: {String.Join(", ", encodedAttention.Select(i => i.ToString().PadLeft(padLength)))}");
		log.AppendLine($"           {String.Join("  ", "".PadLeft(padLength))}");
		log.AppendLine($" Expected: {String.Join(", ", expectedIds.Select(i => i.ToString().PadLeft(padLength)))}");
		log.AppendLine($"  Decoded: {String.Join(", ", expectedEncoded.Select(token => token.PadLeft(padLength)))}");

		using OrtValue? inputOrtValue = OrtValue.CreateTensorValueFromMemory(encodedIds, [1, encodedIds.Length]);
		using OrtValue? encodedAttentionOrtValue = OrtValue.CreateTensorValueFromMemory(encodedAttention, [1, encodedIds.Length]);

		Dictionary<String, OrtValue> inputs = new() {
			{ "input_ids", inputOrtValue },
			{ "attention_mask", encodedAttentionOrtValue },
		};

		using IDisposableReadOnlyCollection<OrtValue>? outputs = _encoderSession.Session.Run(_encoderSession.RunOptions, inputs, _encoderSession.Session.OutputNames);

		OrtValue encoderLastHiddenState = outputs[0];
		OrtValue inputIdsOrtValue = inputs["input_ids"];
		OrtValue inputEncoderAttentionMask = inputs["attention_mask"];

		Dictionary<String, OrtValue> firstRundecoderInputs = new() {
			{ "input_ids", inputIdsOrtValue },
			{ "encoder_attention_mask", inputEncoderAttentionMask },
			{ "encoder_hidden_states", encoderLastHiddenState },
		};
		using IDisposableReadOnlyCollection<OrtValue>? firstRunDecoderOutputs = _decoderSession.Session.Run(_decoderSession.RunOptions, firstRundecoderInputs, _decoderSession.Session.OutputNames);
		using OrtValue logits = firstRunDecoderOutputs[0];
		ReadOnlySpan<Single> logitsData = firstRunDecoderOutputs[0].GetTensorDataAsSpan<Single>();
		Int64 bestToken = TensorPrimitives.IndexOfMax(logitsData.Slice((Int32)_tokenizer.NumberOfTokens * (encodedIds.Length - 1)));
		IDisposableReadOnlyCollection<OrtValue>? lastRunValues = null;
		using OrtValue bestTokenInput = OrtValue.CreateTensorValueFromMemory([bestToken], [1, 1]);
		List<Int64> outputTokens = new(encodedIds.Length) { bestToken };

		OrtValue[] inputValues = [
			inputEncoderAttentionMask,
			bestTokenInput,

			firstRunDecoderOutputs[1],
			firstRunDecoderOutputs[2],
			firstRunDecoderOutputs[3],
			firstRunDecoderOutputs[4],

			firstRunDecoderOutputs[5],
			firstRunDecoderOutputs[6],
			firstRunDecoderOutputs[7],
			firstRunDecoderOutputs[8],

			firstRunDecoderOutputs[9],
			firstRunDecoderOutputs[10],
			firstRunDecoderOutputs[11],
			firstRunDecoderOutputs[12],

			firstRunDecoderOutputs[13],
			firstRunDecoderOutputs[14],
			firstRunDecoderOutputs[15],
			firstRunDecoderOutputs[16],

			firstRunDecoderOutputs[17],
			firstRunDecoderOutputs[18],
			firstRunDecoderOutputs[19],
			firstRunDecoderOutputs[20],

			firstRunDecoderOutputs[21],
			firstRunDecoderOutputs[22],
			firstRunDecoderOutputs[23],
			firstRunDecoderOutputs[24],
		];

		for (Int32 decoderRun = 0; decoderRun < 128 && bestToken != _tokenizer.EndOfSentenceToken; decoderRun++) {
			IDisposableReadOnlyCollection<OrtValue> withPastDecoderOutputs = _decoderWithPastSession.Session.Run(_decoderWithPastSession.RunOptions, _decoderWithPastSession.Session.InputNames, inputValues, _decoderWithPastSession.Session.OutputNames);
			OrtValue withPastlogits = withPastDecoderOutputs[0];
			ReadOnlySpan<Single> withPastLogitsData = withPastlogits.GetTensorDataAsSpan<Single>();
			bestToken = TensorPrimitives.IndexOfMax(withPastLogitsData);
			if (bestToken == _tokenizer.EndOfSentenceToken) {
				withPastDecoderOutputs.Dispose();
				break;
			}

			outputTokens.Add(bestToken);
			lastRunValues?.Dispose();
			lastRunValues = withPastDecoderOutputs;
			bestTokenInput.GetTensorMutableDataAsSpan<Int64>()[0] = bestToken;

			inputValues[2] = lastRunValues[1];
			inputValues[3] = lastRunValues[2];
			inputValues[6] = lastRunValues[3];
			inputValues[7] = lastRunValues[4];
			inputValues[10] = lastRunValues[5];
			inputValues[11] = lastRunValues[6];
			inputValues[14] = lastRunValues[7];
			inputValues[15] = lastRunValues[8];
			inputValues[18] = lastRunValues[9];
			inputValues[19] = lastRunValues[10];
			inputValues[22] = lastRunValues[11];
			inputValues[23] = lastRunValues[12];
		}

		lastRunValues?.Dispose();

		String decodedOutput = _tokenizer.Decode(outputTokens.ToArray());
		Boolean isTestSuccess = String.Equals(decodedOutput, _modelDefinition.SelfTestOutput);

		log.AppendLine($"Test {(isTestSuccess ? "Passed" : "Failed")}");
		log.AppendLine($"Expected output: {_modelDefinition.SelfTestOutput}");
		log.AppendLine($"  Actual output: {decodedOutput}");

		log.AppendLine($"    Index: {String.Join(", ", Enumerable.Range(0, Math.Max(encodedIds.Length, expectedIds.Length)).Select(i => i.ToString().PadLeft(padLength)))}");
		log.AppendLine($"    Token: {String.Join(", ", encoded.Select(token => token.PadLeft(padLength)))}");
		log.AppendLine($" Expected: {String.Join(", ", expectedIds.Select(i => i.ToString().PadLeft(padLength)))}");
		log.AppendLine($"  Decoded: {String.Join(", ", expectedEncoded.Select(token => token.PadLeft(padLength)))}");
		log.AppendLine($"           {String.Join("  ", "".PadLeft(padLength))}");
		log.AppendLine($"   Actual: {String.Join(", ", outputTokens.Select(i => i.ToString().PadLeft(padLength)))}");
		log.AppendLine($"  Decoded: {String.Join(", ", decodedOutput.Split(' ').Select(token => token.PadLeft(padLength)))}");
		return new SelfTestResults() {
			Input = _modelDefinition.SelfTestInput,
			Output = _modelDefinition.SelfTestOutput,
			Success = isTestSuccess,
			Log = log,
		};
	}

	#region IDisposable

	/// <inheritdoc />
	public void Dispose() {
		_encoderSession.Dispose();
		_decoderSession.Dispose();
		_decoderWithPastSession.Dispose();
		_tokenizer.Dispose();
	}

	#endregion
}