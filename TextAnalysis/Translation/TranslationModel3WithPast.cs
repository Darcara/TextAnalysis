namespace TextAnalysis.Translation;

using System.Diagnostics;
using System.Numerics.Tensors;
using System.Text;
using IsoEnums.Iso639;
using Microsoft.ML.OnnxRuntime;
using SentencePieceTokenizer;

[DebuggerDisplay("{ModelDefinition.SourceLanguage}-{ModelDefinition.TargetLanguage}")]
internal class TranslationModel3WithPast : AConfigurationBasedTranslator {
	private readonly OnnxSession _decoderSession;

	public TranslationModel3WithPast(ModelDefinition modelDefinition, OnnxSession encoderSession, OnnxSession decoderSession, OnnxSession decoderWithPastSession, MarianTokenizer tokenizer) : base(modelDefinition, tokenizer, encoderSession, decoderWithPastSession) {
		_decoderSession = decoderSession;
		;
	}

	/// <inheritdoc />
	public override SelfTestResults SelfTest() {
		if (String.IsNullOrEmpty(ModelDefinition.SelfTestInput)) return SelfTestResults.Fail("No Input");
		if (String.IsNullOrEmpty(ModelDefinition.SelfTestOutput)) return SelfTestResults.Fail("No Output");

		StringBuilder log = new();
		log.AppendLine($"Testing 3-part {ModelDefinition.EncoderModelName} with {ModelDefinition.DecoderModelName} and {ModelDefinition.DecoderWithPastModelName} with {EncoderSession.Configuration.OptimizationLevel}");
		(Int64[] encodedIds, Int64[] encodedAttention) = SelfTestLogInput(log, _decoderSession);

		using OrtValue? inputOrtValue = OrtValue.CreateTensorValueFromMemory(encodedIds, [1, encodedIds.Length]);
		using OrtValue? encodedAttentionOrtValue = OrtValue.CreateTensorValueFromMemory(encodedAttention, [1, encodedIds.Length]);

		Dictionary<String, OrtValue> inputs = EncoderSession.CreateStaticInputs(2);
		inputs.Add("input_ids", inputOrtValue);
		inputs.Add("attention_mask", encodedAttentionOrtValue);

		using IDisposableReadOnlyCollection<OrtValue>? encoderOutputs = EncoderSession.Session.Run(EncoderSession.RunOptions, inputs, EncoderSession.Session.OutputNames);
		using OrtValue bestTokenInput = OrtValue.CreateTensorValueFromMemory([Tokenizer.PadToken], [1, 1]);
		OrtValue encoderLastHiddenState = encoderOutputs[0];
		OrtValue inputEncoderAttentionMask = inputs["attention_mask"];

		Dictionary<String, OrtValue> firstRundecoderInputs = _decoderSession.CreateStaticInputs(3);
		firstRundecoderInputs.Add("input_ids", bestTokenInput);
		firstRundecoderInputs.Add("encoder_attention_mask", inputEncoderAttentionMask);
		firstRundecoderInputs.Add("encoder_hidden_states", encoderLastHiddenState);

		using IDisposableReadOnlyCollection<OrtValue>? decoderOutputs = _decoderSession.Session.Run(_decoderSession.RunOptions, firstRundecoderInputs, _decoderSession.Session.OutputNames);

		var outputTokens = SelfTestRunDecoderWithPast(decoderOutputs, bestTokenInput, encodedAttentionOrtValue, encodedIds, encodedAttention, log);

		return SelfTestLogResult(log, outputTokens);
	}

	#region Implementation of AConfigurationBasedTranslator

	/// <inheritdoc />
	protected override void Dispose(Boolean isDisposing) {
		if (isDisposing) _decoderSession.Dispose();
		base.Dispose(isDisposing);
	}

	/// <inheritdoc />
	protected override String InternalTranslate(OrtValue inputOrtValue, OrtValue encodedAttentionOrtValue, Int32 batchSize) {

		Dictionary<String, OrtValue> inputs = EncoderSession.CreateStaticInputs(2);
		inputs.Add("input_ids", inputOrtValue);
		inputs.Add("attention_mask", encodedAttentionOrtValue);

		using IDisposableReadOnlyCollection<OrtValue>? outputs = EncoderSession.Session.Run(EncoderSession.RunOptions, inputs, EncoderSession.Session.OutputNames);

		OrtValue encoderLastHiddenState = outputs[0];

		Int64[] bestTokenValue = new Int64[batchSize];
		Array.Fill(bestTokenValue, Tokenizer.PadToken);
		using OrtValue bestTokenInput = OrtValue.CreateTensorValueFromMemory(bestTokenValue, [batchSize, 1]);
		// using OrtValue bestTokenInput = OrtValue.CreateTensorValueFromMemory([Tokenizer.PadToken], [1, 1]);
		Dictionary<String, OrtValue> firstRundecoderInputs = _decoderSession.CreateStaticInputs(3);
		firstRundecoderInputs.Add("input_ids", bestTokenInput);
		firstRundecoderInputs.Add("encoder_attention_mask", encodedAttentionOrtValue);
		firstRundecoderInputs.Add("encoder_hidden_states", encoderLastHiddenState);

		using IDisposableReadOnlyCollection<OrtValue>? decoderOutputs = _decoderSession.Session.Run(_decoderSession.RunOptions, firstRundecoderInputs, _decoderSession.Session.OutputNames);
		return RunDecoderWithPast(decoderOutputs, bestTokenInput, encodedAttentionOrtValue, batchSize);
	}

	#endregion
}