namespace TextAnalysis;

using IsoEnums.Iso639;

public enum TranslationModelType {
	Undefined = 0,
	
	/// Encoder + Decoder + Decoder-with-past
	Separate3ModelsWithPast,

	/// Merged Encoder+Decoder + Decoder-with-past
	Separate2ModelsWithPast,
	
}

public sealed record ModelDefinition {
	/// <summary>
	/// ISO-639 language designating the source language of this model.
	/// </summary>
	public required Language SourceLanguage { get; init; }

	/// <summary>
	/// ISO-639 language designating the target language of this model.
	/// </summary>
	public required Language TargetLanguage { get; init; }
	
	public required TranslationModelType Type { get; init; }

	public String SourceTokenizer { get; init; } = "source.spm";
	public String VocabularyTokenToIdMap { get; init; } = "vocab.json";
	
	public required Int32 SourceMaxTokens { get; init; }

	public String? ModelDirectoryOverride { get; init; }
	public String ModelDirectory => ModelDirectoryOverride ?? $"{SourceLanguage.Get2Code()}-{TargetLanguage.Get2Code()}";

	public String? EncoderModelNameOverride { get; init; }
	public String EncoderModelName => EncoderModelNameOverride ?? "encoder_model.onnx";
	public String? DecoderModelNameOverride { get; init; }
	public String DecoderModelName => DecoderModelNameOverride ?? "decoder_model.onnx";
	public String? DecoderWithPasthModelNameOverride { get; init; }
	public String DecoderWithPastModelName => DecoderWithPasthModelNameOverride ?? "decoder_with_past_model.onnx";

	/// <summary>
	/// The input sentence to test that the model and its weights have loaded correctly
	/// </summary>
	public String? SelfTestInput { get; init; }

	/// <summary>
	/// The output sentence to test that the model and its weights have loaded correctly
	/// </summary>
	public String? SelfTestOutput { get; init; }
	
	/// <summary>
	/// Used to test the performance of a model in a variety of confugurations.
	/// Make sure it is at least 8*512 = <b>4096 tokens</b> long.
	/// </summary>
	public String? PerformanceTestInput { get; init; }

	public ModelDefinition WithExternalWeights() {
		return this with {
			EncoderModelNameOverride = Path.ChangeExtension(EncoderModelName, ".clear" + Path.GetExtension(EncoderModelName)),
			DecoderModelNameOverride = Path.ChangeExtension(DecoderModelName, ".clear" + Path.GetExtension(DecoderModelName)),
			DecoderWithPasthModelNameOverride = Path.ChangeExtension(DecoderWithPastModelName, ".clear" + Path.GetExtension(DecoderWithPastModelName)),
		};
	}
}