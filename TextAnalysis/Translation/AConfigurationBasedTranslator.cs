namespace TextAnalysis.Translation;

using System.Buffers;
using System.Diagnostics;
using System.Net;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Text;
using IsoEnums.Iso639;
using Microsoft.ML.OnnxRuntime;
using Neco.Common;
using Neco.Common.Extensions;
using SentencePieceTokenizer;

public abstract class AConfigurationBasedTranslator : ITranslator {
	protected readonly ModelDefinition ModelDefinition;
	protected readonly MarianTokenizer Tokenizer;
	protected readonly OnnxSession EncoderSession;
	protected readonly OnnxSession DecoderWithPastSession;

	protected AConfigurationBasedTranslator(ModelDefinition modelDefinition, MarianTokenizer tokenizer, OnnxSession encoderSession, OnnxSession decoderWithPastSession) {
		ModelDefinition = modelDefinition;
		EncoderSession = encoderSession;
		DecoderWithPastSession = decoderWithPastSession;
		Tokenizer = tokenizer;
	}

	protected (Int64[] encodedIds, Int64[] encodedAttention) SelfTestLogInput(StringBuilder log, params IEnumerable<OnnxSession> sessions) {
		// only required for logging
		String[] encoded = Tokenizer.EncodeToStrings(ModelDefinition.SelfTestInput);
		String[] expectedEncoded = Tokenizer.EncodeToStrings(ModelDefinition.SelfTestOutput);

		Int64[] encodedIds = Tokenizer.EncodeToIds(ModelDefinition.SelfTestInput, suffixIds: [Tokenizer.EndOfSentenceToken]);
		Int64[] expectedIds = Tokenizer.EncodeToIds(ModelDefinition.SelfTestOutput);

		Int64[] encodedAttention = new Int64[encodedIds.Length];
		Array.Fill(encodedAttention, 1);

		Int32 padLength = encoded.Max(token => token.Length);
		IEnumerable<OnnxSession> logSessions = [EncoderSession, ..sessions, DecoderWithPastSession];
		foreach (var session in logSessions) {
			log.AppendLine($"{session.Name}");
			foreach (KeyValuePair<String, NodeMetadata> input in session.Session.InputMetadata)
				log.AppendLine($"  Input : {input.Key} = {input.Value.ElementType.GetName()}[{String.Join(", ", input.Value.Dimensions.Select((dim, idx) => dim == -1 ? input.Value.SymbolicDimensions[idx] : dim.ToString()))}] as {input.Value.OnnxValueType}");
			foreach (KeyValuePair<String, NodeMetadata> output in session.Session.OutputMetadata)
				log.AppendLine($"  Output: {output.Key} = {output.Value.ElementType.GetName()}[{String.Join(", ", output.Value.Dimensions.Select((dim, idx) => dim == -1 ? output.Value.SymbolicDimensions[idx] : dim.ToString()))}] as {output.Value.OnnxValueType}");
		}

		log.AppendLine();
		log.AppendLine($"    Index: {String.Join(", ", Enumerable.Range(0, Math.Max(encodedIds.Length, expectedIds.Length)).Select(i => i.ToString().PadLeft(padLength)))}");
		log.AppendLine($"    Token: {String.Join(", ", encoded.Select(token => token.PadLeft(padLength)))}");
		log.AppendLine($"  Encoded: {String.Join(", ", encodedIds.Select(i => i.ToString().PadLeft(padLength)))}");
		log.AppendLine($"Attention: {String.Join(", ", encodedAttention.Select(i => i.ToString().PadLeft(padLength)))}");
		log.AppendLine($"           {String.Join("  ", "".PadLeft(padLength))}");
		log.AppendLine($" Expected: {String.Join(", ", expectedIds.Select(i => i.ToString().PadLeft(padLength)))}");
		log.AppendLine($"  Decoded: {String.Join(", ", expectedEncoded.Select(token => token.PadLeft(padLength)))}");

		return (encodedIds, encodedAttention);
	}

	protected void SelfTestLogProgress(StringBuilder log, Int32 round, IEnumerable<Int64> outputTokens, IEnumerable<Int64> inputs, IEnumerable<Int64> attention) {
		String[] decoded = outputTokens.Select(t => Tokenizer.Decode(t)).ToArray();
		Int32 padLength = Math.Max(3, decoded.Max(str => str.Length));
		log.AppendLine($" Round {round,2}: {String.Join(", ", outputTokens.Select(i => i.ToString().PadLeft(padLength)))}");
		log.AppendLine($"  Decoded: {String.Join(", ", decoded.Select(token => token.PadLeft(padLength)))}");
		if (inputs.Any())
			log.AppendLine($"  Encoded: {String.Join(", ", inputs.Select(i => i.ToString().PadLeft(padLength)))}");
		if (attention.Any())
			log.AppendLine($"Attention: {String.Join(", ", attention.Select(i => i.ToString().PadLeft(padLength)))}");
	}

	protected SelfTestResults SelfTestLogResult(StringBuilder log, IEnumerable<Int64> outputTokens) {
		String decodedOutput = Tokenizer.Decode(outputTokens.ToArray());
		Boolean isTestSuccess = String.Equals(decodedOutput, ModelDefinition.SelfTestOutput);

		String[] encoded = Tokenizer.EncodeToStrings(ModelDefinition.SelfTestInput);
		String[] expectedEncoded = Tokenizer.EncodeToStrings(ModelDefinition.SelfTestOutput);

		Int64[] encodedIds = Tokenizer.EncodeToIds(ModelDefinition.SelfTestInput, suffixIds: [Tokenizer.EndOfSentenceToken, Tokenizer.PadToken]);
		Int64[] expectedIds = Tokenizer.EncodeToIds(ModelDefinition.SelfTestOutput);

		Int32 padLength = encoded.Max(token => token.Length);
		log.AppendLine($"Test {(isTestSuccess ? "Passed" : "Failed")}");
		log.AppendLine($"Expected output: {ModelDefinition.SelfTestOutput}");
		log.AppendLine($"  Actual output: {decodedOutput}");

		log.AppendLine($"    Index: {String.Join(", ", Enumerable.Range(0, Math.Max(encodedIds.Length, expectedIds.Length)).Select(i => i.ToString().PadLeft(padLength)))}");
		log.AppendLine($"    Token: {String.Join(", ", encoded.Select(token => token.PadLeft(padLength)))}");
		log.AppendLine($" Expected: {String.Join(", ", expectedIds.Select(i => i.ToString().PadLeft(padLength)))}");
		log.AppendLine($"  Decoded: {String.Join(", ", expectedEncoded.Select(token => token.PadLeft(padLength)))}");
		log.AppendLine($"           {String.Join("  ", "".PadLeft(padLength))}");
		log.AppendLine($"   Actual: {String.Join(", ", outputTokens.Select(i => i.ToString().PadLeft(padLength)))}");
		log.AppendLine($"  Decoded: {String.Join(", ", decodedOutput.Split(' ').Select(token => token.PadLeft(padLength)))}");
		return new SelfTestResults(isTestSuccess, ModelDefinition.SelfTestInput, ModelDefinition.SelfTestOutput, log);
	}

	protected List<Int64> SelfTestRunDecoderWithPast(IDisposableReadOnlyCollection<OrtValue> decoderOutputs, OrtValue bestTokenInput, OrtValue encodedAttentionOrtValue, Int64[] encodedIds, Int64[] encodedAttention, StringBuilder log) {
		using OrtValue logits = decoderOutputs[0];
		ReadOnlySpan<Single> logitsData = decoderOutputs[0].GetTensorDataAsSpan<Single>();
		Int64 bestToken = TensorPrimitives.IndexOfMax(logitsData);
		IDisposableReadOnlyCollection<OrtValue>? lastRunValues = null;
		bestTokenInput.GetTensorMutableDataAsSpan<Int64>()[0] = bestToken;
		List<Int64> outputTokens = new(encodedIds.Length) { bestToken };

		SelfTestLogProgress(log, 0, outputTokens, encodedIds, encodedAttention);

		OrtValue[] inputValues = DecoderWithPastSession.CreateStaticInputArray(26);
		inputValues[0] = encodedAttentionOrtValue;
		inputValues[1] = bestTokenInput;
		for (int i = 1; i < decoderOutputs.Count; ++i)
			inputValues[i + 1] = decoderOutputs[i];

		for (Int32 decoderRun = 0; decoderRun < 128 && bestToken != Tokenizer.EndOfSentenceToken; decoderRun++) {
			IDisposableReadOnlyCollection<OrtValue> withPastDecoderOutputs = DecoderWithPastSession.Session.Run(DecoderWithPastSession.RunOptions, DecoderWithPastSession.Session.InputNames, inputValues, DecoderWithPastSession.Session.OutputNames);
			OrtValue withPastlogits = withPastDecoderOutputs[0];
			ReadOnlySpan<Single> withPastLogitsData = withPastlogits.GetTensorDataAsSpan<Single>();
			bestToken = TensorPrimitives.IndexOfMax(withPastLogitsData);
			if (bestToken == Tokenizer.EndOfSentenceToken) {
				withPastDecoderOutputs.Dispose();
				break;
			}

			outputTokens.Add(bestToken);

			SelfTestLogProgress(log, decoderRun + 1, outputTokens, [bestToken], encodedAttention);

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
		return outputTokens;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	protected String RunDecoderWithPast(IDisposableReadOnlyCollection<OrtValue> decoderOutputs, OrtValue bestTokenInput, OrtValue encodedAttentionOrtValue, Int32 batchSize) {
		if (batchSize > 1) return RunDecoderWithPastBatched(decoderOutputs, bestTokenInput, encodedAttentionOrtValue, batchSize);
		ReadOnlySpan<Single> logitsData = decoderOutputs[0].GetTensorDataAsSpan<Single>();
		Int64 bestToken = TensorPrimitives.IndexOfMax(logitsData);
		IDisposableReadOnlyCollection<OrtValue>? lastRunValues = null;
		bestTokenInput.GetTensorMutableDataAsSpan<Int64>()[0] = bestToken;
		Int64[] outputTokens = ArrayPool<Int64>.Shared.Rent(512);
		outputTokens[0] = bestToken;
		Int32 outputTokenIndex = 1;

		OrtValue[] inputValues = DecoderWithPastSession.CreateStaticInputArray(26);
		inputValues[0] = encodedAttentionOrtValue;
		inputValues[1] = bestTokenInput;
		for (int i = 1; i < decoderOutputs.Count; ++i)
			inputValues[i + 1] = decoderOutputs[i];

		for (Int32 decoderRun = 0; decoderRun < 510 && bestToken != Tokenizer.EndOfSentenceToken; decoderRun++) {
			IDisposableReadOnlyCollection<OrtValue> withPastDecoderOutputs = DecoderWithPastSession.Session.Run(DecoderWithPastSession.RunOptions, DecoderWithPastSession.Session.InputNames, inputValues, DecoderWithPastSession.Session.OutputNames);
			OrtValue withPastlogits = withPastDecoderOutputs[0];
			ReadOnlySpan<Single> withPastLogitsData = withPastlogits.GetTensorDataAsSpan<Single>();
			bestToken = TensorPrimitives.IndexOfMax(withPastLogitsData);
			if (bestToken == Tokenizer.EndOfSentenceToken) {
				withPastDecoderOutputs.Dispose();
				break;
			}

			if (bestToken != Tokenizer.PadToken)
				outputTokens[outputTokenIndex++] = bestToken;

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

		String result = Tokenizer.Decode(new ReadOnlySpan<Int64>(outputTokens, 0, outputTokenIndex));
		ArrayPool<Int64>.Shared.Return(outputTokens);
		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private String RunDecoderWithPastBatched(IDisposableReadOnlyCollection<OrtValue> decoderOutputs, OrtValue bestTokenInput, OrtValue encodedAttentionOrtValue, Int32 batchSize) {
		ReadOnlySpan<Single> logitsData = decoderOutputs[0].GetTensorDataAsSpan<Single>();
		Int64[] outputTokens = ArrayPool<Int64>.Shared.Rent(512 * batchSize);
		Int64[] bestTokens = new Int64[batchSize];
		for (int i = 0; i < batchSize; i++) {
			bestTokens[i] = TensorPrimitives.IndexOfMax(logitsData.Slice(i * Tokenizer.NumberOfTokens, Tokenizer.NumberOfTokens));
			outputTokens[i * 512] = bestTokens[i];
		}

		Span<Int64> bestTokenSpan = bestTokenInput.GetTensorMutableDataAsSpan<Int64>();
		new ReadOnlySpan<Int64>(bestTokens).CopyTo(bestTokenSpan);

		Int32 outputTokenIndex = 1;

		OrtValue[] inputValues = DecoderWithPastSession.CreateStaticInputArray(26);
		inputValues[0] = encodedAttentionOrtValue;
		inputValues[1] = bestTokenInput;
		for (int i = 1; i < decoderOutputs.Count; ++i)
			inputValues[i + 1] = decoderOutputs[i];

		IDisposableReadOnlyCollection<OrtValue>? lastRunValues = null;
		Int32[] outputLengths = new Int32[batchSize];
		Array.Fill(outputLengths, 1);

		for (Int32 decoderRun = 0; decoderRun < 510 /*&& bestToken != Tokenizer.EndOfSentenceToken*/; decoderRun++) {
			IDisposableReadOnlyCollection<OrtValue> withPastDecoderOutputs = DecoderWithPastSession.Session.Run(DecoderWithPastSession.RunOptions, DecoderWithPastSession.Session.InputNames, inputValues, DecoderWithPastSession.Session.OutputNames);
			OrtValue withPastlogits = withPastDecoderOutputs[0];
			ReadOnlySpan<Single> withPastLogitsData = withPastlogits.GetTensorDataAsSpan<Single>();
			// bestToken = TensorPrimitives.IndexOfMax(withPastLogitsData);
			Boolean allBatchesFinished = true;
			for (int i = 0; i < batchSize; i++) {
				bestTokens[i] = TensorPrimitives.IndexOfMax(withPastLogitsData.Slice(i * Tokenizer.NumberOfTokens, Tokenizer.NumberOfTokens));
				outputTokens[i * 512 + outputTokenIndex] = bestTokens[i];
				if (bestTokens[i] != Tokenizer.EndOfSentenceToken && outputTokenIndex == outputLengths[i]) {
					outputLengths[i]++;
					allBatchesFinished = false;
				}
			}

			if (allBatchesFinished) {
				withPastDecoderOutputs.Dispose();
				break;
			}

			// if (bestToken != Tokenizer.PadToken)
			// outputTokens[outputTokenIndex++] = bestToken;
			++outputTokenIndex;

			lastRunValues?.Dispose();
			lastRunValues = withPastDecoderOutputs;
			// bestTokenInput.GetTensorMutableDataAsSpan<Int64>()[0] = bestToken;
			bestTokenSpan = bestTokenInput.GetTensorMutableDataAsSpan<Int64>();
			bestTokens.CopyTo(bestTokenSpan);

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

		// String result = Tokenizer.Decode(new ReadOnlySpan<Int64>(outputTokens, 0, outputTokenIndex));
		String result = String.Empty;
		for (int i = 0; i < batchSize; i++) {
			String s = Tokenizer.Decode(new ReadOnlySpan<Int64>(outputTokens, i * 512, outputLengths[i]));
			result = (result + "\n" + s).Trim();
		}

		ArrayPool<Int64>.Shared.Return(outputTokens);
		return result;
	}

	#region Implementation of ITranslator

	/// <inheritdoc />
	public Language[] From => [ModelDefinition.SourceLanguage];

	/// <inheritdoc />
	public Language[] To => [ModelDefinition.TargetLanguage];

	/// <inheritdoc />
	public virtual Boolean CanBeUsedConcurrently => EncoderSession.Configuration.ExecutionProvider == ExecutionProvider.CPU;

	/// <inheritdoc />
	public virtual XXX CanTranslate(Language from, Language to) {
		XXX code = XXX.Unknown;
		if (!From.Contains(from))
			code |= XXX.UnknownSourceLanguage;
		if (!To.Contains(to))
			code |= XXX.UnknownTargetLanguage;

		if (code == XXX.Unknown) return XXX.Available;
		return code;
	}

	/// <inheritdoc />
	public IEnumerable<String> Translate(String[] text, Language from, Language to) => text.Select(txt => Translate(txt, from, to));

	public IEnumerable<String> Translate3(ReadOnlySpan<Byte> utf8Bytes, ReadOnlySpan<Int32> splits, Language from, Language to) {
		List<String> result = new List<String>(splits.Length);
		Int32 offset = 0;
		for (int i = 0; i < splits.Length; i++) {
			Int32 split = splits[i];
			ReadOnlySpan<Byte> currentSlice = utf8Bytes.Slice(offset, split - offset);
			if (currentSlice.Length == 0) continue;
			Int64[] encodedIds = Tokenizer.EncodeToIds(currentSlice, suffixIds: [Tokenizer.EndOfSentenceToken]);
			Int64[] encodedAttention = new Int64[encodedIds.Length];

			Array.Fill(encodedAttention, 1);

			using OrtValue? inputOrtValue = OrtValue.CreateTensorValueFromMemory(encodedIds, [1, encodedIds.Length]);
			using OrtValue? encodedAttentionOrtValue = OrtValue.CreateTensorValueFromMemory(encodedAttention, [1, encodedIds.Length]);

			String translated = InternalTranslate(inputOrtValue, encodedAttentionOrtValue, 1);
			// String translated = Translate(currentSlice, from, to);
			result.Add(translated);
			offset = split;
		}

		return result;
	}

	/// <inheritdoc />
	public IEnumerable<String> Translate2(ReadOnlySpan<Byte> utf8Bytes, ReadOnlySpan<Int32> splits, Language from, Language to) {
		List<String> result = new List<String>(splits.Length);
		Int32 alreadyTranslatedTokensOffset = 0;
		Int32 splitOffset = 0;
		(Int64[] ids, TokenSpan[] tokens) = Tokenizer.EncodeToSpans(utf8Bytes);

		while (true) {
			Int32 nextTokenOffset = alreadyTranslatedTokensOffset;
			Int32 nextSplitOffset = splitOffset;
			for (int i = splitOffset; i < splits.Length; i++) {
				Int32 split = splits[i];
				Int32 idx = tokens.FindIndex(nextTokenOffset, t => t.End == split);
				Int32 tokensToMaybeTranslate = idx - alreadyTranslatedTokensOffset + 1 + 1;
				if (tokensToMaybeTranslate <= 512) {
					nextTokenOffset = idx;
					nextSplitOffset = i + 1;
					break;
				} else
					break;
			}

			Int32 tokensToTranslate = nextTokenOffset - alreadyTranslatedTokensOffset + 1;
			if (tokensToTranslate == 1) break;

			Int64[] encodedIds = ArrayPool<Int64>.Shared.Rent(tokensToTranslate + 1);
			Int64[] encodedAttention = ArrayPool<Int64>.Shared.Rent(tokensToTranslate + 1);
			ids.AsSpan().Slice(alreadyTranslatedTokensOffset, tokensToTranslate).CopyTo(encodedIds);
			encodedIds[tokensToTranslate] = Tokenizer.EndOfSentenceToken;
			Array.Fill(encodedIds, Tokenizer.PadToken, tokensToTranslate + 1, encodedIds.Length - (tokensToTranslate + 1));

			Array.Fill(encodedAttention, 1, 0, tokensToTranslate + 1);
			Array.Fill(encodedAttention, 0, tokensToTranslate + 1, encodedAttention.Length - (tokensToTranslate + 1));
			Console.WriteLine($"Encoded: {String.Join(" ", encodedIds)}");
			Console.WriteLine($"Encoded: {String.Join(" ", encodedAttention)}");

			using OrtValue? inputOrtValue = OrtValue.CreateTensorValueFromMemory(encodedIds, [1, encodedIds.Length]);
			using OrtValue? encodedAttentionOrtValue = OrtValue.CreateTensorValueFromMemory(encodedAttention, [1, encodedAttention.Length]);

			String translated = InternalTranslate(inputOrtValue, encodedAttentionOrtValue, 1);
			result.Add(translated);

			ArrayPool<Int64>.Shared.Return(encodedIds);
			ArrayPool<Int64>.Shared.Return(encodedAttention);

			alreadyTranslatedTokensOffset = nextTokenOffset + 1;
			splitOffset = nextSplitOffset;
		}

		return result;
	}

	public IEnumerable<String> Translate(ReadOnlySpan<Byte> utf8Bytes, ReadOnlySpan<Int32> splits, Language from, Language to) {
		Int32 batchSize = EncoderSession.Configuration.Batching.BatchSize;

		if (batchSize > 1) return TranslateBatch(utf8Bytes, splits, batchSize);

		List<String> result = new List<String>(splits.Length);
		Int32 offset = 0;
		// Encode once and copy relevant part is faster than encoding each split/slice
		(Int64[] ids, TokenSpan[] tokens) = Tokenizer.EncodeToSpans(utf8Bytes);

		// Multiple sentences cannot be "batched" into one slice, even if they are below 512 tokens, as the translation result will be of very low quality.


		for (int i = 0; i < splits.Length; i++) {
			Int32 split = splits[i];
			Int32 idx = tokens.FindIndex(offset, t => t.End == split);

			Int64[] encodedIds = ArrayPool<Int64>.Shared.Rent(idx - offset + 1 + 1);
			Int64[] encodedAttention = ArrayPool<Int64>.Shared.Rent(idx - offset + 1 + 1);
			ids.AsSpan().Slice(offset, idx - offset + 1).CopyTo(encodedIds);
			encodedIds[idx - offset + 1] = Tokenizer.EndOfSentenceToken;

			Array.Fill(encodedAttention, 1, 0, idx - offset + 1 + 1);
			Array.Fill(encodedAttention, 0, idx - offset + 1 + 1, encodedIds.Length - (idx - offset + 1 + 1));


			using OrtValue? inputOrtValue = OrtValue.CreateTensorValueFromMemory(encodedIds, [batchSize, encodedIds.Length]);
			using OrtValue? encodedAttentionOrtValue = OrtValue.CreateTensorValueFromMemory(encodedAttention, [batchSize, encodedIds.Length]);

			String translated = InternalTranslate(inputOrtValue, encodedAttentionOrtValue, batchSize);
			result.Add(translated);
			offset = idx + 1;

			ArrayPool<Int64>.Shared.Return(encodedIds);
			ArrayPool<Int64>.Shared.Return(encodedAttention);
		}

		return result;
	}

	private IEnumerable<String> TranslateBatch(ReadOnlySpan<Byte> utf8Bytes, ReadOnlySpan<Int32> splits, Int32 batchSize) {
		List<String> result = new List<String>(splits.Length);
		Int32 alreadyTranslatedTokensOffset = 0;
		Int32 splitOffset = 0;
		(Int64[] ids, TokenSpan[] tokens) = Tokenizer.EncodeToSpans(utf8Bytes);

		while (true) {
			Int32 nextTokenOffset = alreadyTranslatedTokensOffset;
			Int32 nextSplitOffset = splitOffset;
			Int32 largestSplit = 0;
			ValueTuple<Int32, Int32>[] tokenSlices = new ValueTuple<Int32, Int32>[batchSize];

			// for (int i = splitOffset; i < splits.Length; i++) {
			for (int i = 0; i < batchSize; i++) {
				if (nextSplitOffset >= splits.Length) {
					tokenSlices[i] = (0, 0);
					continue;
				}

				// ++numBatches; make BatchModels
				Int32 split = splits[nextSplitOffset];
				Int32 idx = tokens.FindIndex(nextTokenOffset, t => t.End == split);
				Int32 tokensToMaybeTranslate = idx - nextTokenOffset + 1;
				largestSplit = Math.Max(tokensToMaybeTranslate, largestSplit);
				// Console.WriteLine($"From {nextTokenOffset} -- {idx} == {tokensToMaybeTranslate} --- MAX={largestSplit}");
				tokenSlices[i] = (nextTokenOffset, tokensToMaybeTranslate);
				nextTokenOffset = idx + 1;
				nextSplitOffset++;
			}

			// Int32 tokensToTranslate = nextTokenOffset - alreadyTranslatedTokensOffset ;
			if (largestSplit <= 1) break;

			Int32 sizeOfOneBatch = largestSplit + 1;
			Int64[] encodedIds = ArrayPool<Int64>.Shared.Rent(sizeOfOneBatch * batchSize);
			Int64[] encodedAttention = ArrayPool<Int64>.Shared.Rent(sizeOfOneBatch * batchSize);
			for (int i = 0; i < batchSize; i++) {
				Int32 idOffset = tokenSlices[i].Item1;
				Int32 tokensToTranslate = tokenSlices[i].Item2;
				Span<Int64> encodedIdSlice = encodedIds.AsSpan(i * sizeOfOneBatch, sizeOfOneBatch);
				Span<Int64> encodedAttentionSlice = encodedAttention.AsSpan(i * sizeOfOneBatch, sizeOfOneBatch);
				if (tokensToTranslate > 0) {
					ids.AsSpan().Slice(idOffset, tokensToTranslate).CopyTo(encodedIdSlice);
					encodedIdSlice[tokensToTranslate] = Tokenizer.EndOfSentenceToken;
					encodedIdSlice.Slice(tokensToTranslate + 1).Fill(Tokenizer.PadToken);
					encodedAttentionSlice.Slice(0, tokensToTranslate + 1).Fill(1L);
					encodedAttentionSlice.Slice(tokensToTranslate + 1).Fill(0L);
				} else {
					encodedIdSlice.Fill(Tokenizer.PadToken);
					encodedAttentionSlice.Fill(0L);
				}

				// Console.WriteLine($"Encoded: {String.Join(" ", encodedIdSlice.ToArray())}");
				// Console.WriteLine($"Encoded: {String.Join(" ", encodedAttentionSlice.ToArray())}");
			}

			using OrtValue? inputOrtValue = OrtValue.CreateTensorValueFromMemory(encodedIds, [batchSize, sizeOfOneBatch]);
			using OrtValue? encodedAttentionOrtValue = OrtValue.CreateTensorValueFromMemory(encodedAttention, [batchSize, sizeOfOneBatch]);

			String translated = InternalTranslate(inputOrtValue, encodedAttentionOrtValue, batchSize);
			// Console.WriteLine($"Translated: {translated}");
			result.Add(translated);

			ArrayPool<Int64>.Shared.Return(encodedIds);
			ArrayPool<Int64>.Shared.Return(encodedAttention);

			alreadyTranslatedTokensOffset = nextTokenOffset;
			splitOffset = nextSplitOffset;
		}

		return result;
	}

	/// <inheritdoc />
	public abstract SelfTestResults SelfTest();

	/// <inheritdoc />
	public String Translate(ReadOnlySpan<Char> text, Language from, Language to) {
		Int32 requiredBytes = Encoding.UTF8.GetByteCount(text);
		Byte[] buffer = ArrayPool<Byte>.Shared.Rent(requiredBytes);
		Span<Byte> span = buffer.AsSpan();
		try {
			Int32 bytesWritten = MagicNumbers.Utf8NoBom.GetBytes(text, span);
			return Translate(span.Slice(0, bytesWritten), from, to);
		}
		finally {
			ArrayPool<Byte>.Shared.Return(buffer);
		}
	}

	/// <inheritdoc />
	public String Translate(ReadOnlySpan<Byte> utf8Bytes, Language from, Language to) {
		if (utf8Bytes.Length == 0) return String.Empty;

		Int64[] encodedIds = Tokenizer.EncodeToIds(utf8Bytes, suffixIds: [Tokenizer.EndOfSentenceToken]);
		Console.WriteLine($"Encoded: {String.Join(" ", encodedIds)}");
		if (encodedIds.Length > 511) throw new ArgumentOutOfRangeException(nameof(utf8Bytes), encodedIds.Length, $"Input string tokenized into {encodedIds.Length} tokens. Max allowed tokens: 511.");

		Int64[] encodedAttention = new Int64[encodedIds.Length];
		Array.Fill(encodedAttention, 1);

		using OrtValue? inputOrtValue = OrtValue.CreateTensorValueFromMemory(encodedIds, [1, encodedIds.Length]);
		using OrtValue? encodedAttentionOrtValue = OrtValue.CreateTensorValueFromMemory(encodedAttention, [1, encodedIds.Length]);

		return InternalTranslate(inputOrtValue, encodedAttentionOrtValue, 1);
	}

	#endregion

	protected abstract String InternalTranslate(OrtValue inputOrtValue, OrtValue encodedAttentionOrtValue, Int32 batchSize);

	#region IDisposable

	protected virtual void Dispose(bool disposing) {
		if (disposing) {
			EncoderSession.Dispose();
			DecoderWithPastSession.Dispose();
			Tokenizer.Dispose();
		}
	}

	/// <inheritdoc />
	public void Dispose() {
		Dispose(true);
		GC.SuppressFinalize(this);
	}

	#endregion
}