namespace TextAnalysis;

using System.Collections.Frozen;
using System.IO.Hashing;
using System.Text.Json;
using Neco.Common.Data.Hash;
using Neco.Common.Extensions;
using Onnx;
using ProtoBuf;
using TextAnalysis.ProtobufModels;

public record InitializerDefinition(String File, String Type, Int64[]? Shape);

public record InitializerConfig(Dictionary<String, String> Initializers, Int64 TotalSize);

public class OnnxModelInspector {
	public static void Inspect(String filename) {
		ModelProto model = LoadModel(filename) ?? throw new ArgumentException($"Unable to load onnx model from {filename}", nameof(filename));
		Console.WriteLine($"Model {filename} has {model.Graph.Initializers.Count} initializers, of wich {model.Graph.Initializers.Select(i => i.Name).Distinct().Count()} are unique");
	}

	public static void Compare(String filename1, String filename2) {
		ModelProto model1 = LoadModel(filename1) ?? throw new ArgumentException($"Unable to load onnx model from {filename1}", nameof(filename1));
		ModelProto model2 = LoadModel(filename2) ?? throw new ArgumentException($"Unable to load onnx model from {filename2}", nameof(filename2));
		FrozenSet<String> initializers1 = model1.Graph.Initializers.Select(i => i.Name).Distinct().ToFrozenSet();
		FrozenSet<String> initializers2 = model2.Graph.Initializers.Select(i => i.Name).Distinct().ToFrozenSet();
		Int64 initializerSize1 = model1.Graph.Initializers.Select(i => i.CalculateSize()).Sum();
		Int64 initializerSize2 = model2.Graph.Initializers.Select(i => i.CalculateSize()).Sum();
		Console.WriteLine($"Model {filename1} has {model1.Graph.Initializers.Count} ({initializerSize1.ToFileSize()}) initializers, of wich {initializers1.Count} are unique");
		Console.WriteLine($"Model {filename2} has {model2.Graph.Initializers.Count} ({initializerSize2.ToFileSize()}) initializers, of wich {initializers2.Count} are unique");
		Console.WriteLine($"Model {filename1} has {initializers1.Except(initializers2).Count()} initializers, that {filename2} does not have");
		Console.WriteLine($"Model {filename2} has {initializers2.Except(initializers1).Count()} initializers, that {filename1} does not have");
		FrozenSet<String> sharedInitializerNames = initializers1.Intersect(initializers2).ToFrozenSet();
		Console.WriteLine($"Models share {sharedInitializerNames.Count} initializers by name");
		Console.WriteLine($"Models have a total of {initializers1.Count + initializers2.Count} initializers of wich {initializers1.Union(initializers2).Count()} are unique");

		Console.WriteLine($"Model {filename1} ops:" + String.Join(", ", model1.Graph.Nodes.GroupBy(n => n.OpType).OrderBy(grp => grp.Key).Select(grp => $"{grp.Key}x{grp.Count()}")));
		Console.WriteLine($"Model {filename2} ops:" + String.Join(", ", model1.Graph.Nodes.GroupBy(n => n.OpType).OrderBy(grp => grp.Key).Select(grp => $"{grp.Key}x{grp.Count()}")));

		Int64 totalSharedSize = 0;
		Int64 totalShared = 0;
		foreach (String sharedInitializerName in sharedInitializerNames) {
			TensorProto sharedInitializer1 = model1.Graph.Initializers.First(i => i.Name == sharedInitializerName);
			TensorProto sharedInitializer2 = model2.Graph.Initializers.First(i => i.Name == sharedInitializerName);
			Int64 size1 = sharedInitializer1.CalculateSize();
			Int64 size2 = sharedInitializer2.CalculateSize();
			if (size1 != size2) {
				Console.WriteLine($"Shared intializer {sharedInitializerName} has different sizes in models: {size1.ToFileSize()} vs {size2.ToFileSize()}");
				continue;
			}

			var data1 = sharedInitializer1.RawData.AsSpan();
			var data2 = sharedInitializer1.RawData;
			if (!data1.SequenceEqual(data2)) {
				Console.WriteLine($"Shared intializer {sharedInitializerName} have same size ({size1.ToFileSize()}) but differing content");
				continue;
			}

			totalSharedSize += size1;
			++totalShared;

			// Console.WriteLine($"{sharedInitializerName} are equal by data ({size1.ToFileSize()}) and dimensions {String.Join("x", sharedInitializer1.Dims??[])} -- {String.Join("x", sharedInitializer2.Dims??[])}");
		}

		Console.WriteLine($"Models share {totalShared} initializers by name with {totalSharedSize.ToFileSize()}");

		HashSet<String> duplicatedInitializersSelf1 = new();
		HashSet<String> duplicatedInitializersSelf2 = new();
		HashSet<String> duplicatedInitializersOther = new();
		Int64 selfByValueSharedSize1 = 0;
		Int64 selfByValueSharedSize2 = 0;
		Int64 totalByValueSharedSize = 0;
		for (Int32 index = 0; index < model1.Graph.Initializers.Count; index++) {
			TensorProto initializer1 = model1.Graph.Initializers[index];
			foreach ((String? name, Int64 size) in Compare(initializer1, model1.Graph.Initializers, index + 1)) {
				duplicatedInitializersSelf1.Add(initializer1.Name);
				if (duplicatedInitializersSelf1.Add(name)) {
					selfByValueSharedSize1 += size;
				}
			}

			foreach ((String? name, Int64 size) in Compare(initializer1, model2.Graph.Initializers, 0)) {
				if (duplicatedInitializersOther.Add(name)) {
					totalByValueSharedSize += size;
					// Console.WriteLine($"{initializer1.Name} and {name} are equal by data ({size.ToFileSize()})");
				}
			}
		}

		for (Int32 index = 0; index < model2.Graph.Initializers.Count; index++) {
			TensorProto initializer2 = model2.Graph.Initializers[index];
			foreach ((String? name, Int64 size) in Compare(initializer2, model2.Graph.Initializers, index + 1)) {
				duplicatedInitializersSelf2.Add(initializer2.Name);
				if (duplicatedInitializersSelf2.Add(name)) {
					selfByValueSharedSize2 += size;
				}
			}
		}

		Console.WriteLine($"Model {filename1} self shares {duplicatedInitializersSelf1.Count} initializers with {selfByValueSharedSize1.ToFileSize()}");
		Console.WriteLine($"Model {filename2} self shares {duplicatedInitializersSelf2.Count} initializers with {selfByValueSharedSize2.ToFileSize()}");
		Console.WriteLine($"Models share {duplicatedInitializersOther.Count} initializers by value with {totalByValueSharedSize.ToFileSize()}");
	}

	private static IEnumerable<(String name, Int64 size)> Compare(TensorProto baseInitializer, List<TensorProto> initializers, Int32 startIndex) {
		for (Int32 i = startIndex; i < initializers.Count; i++) {
			TensorProto sameGraphInitializer = initializers[i];
			if (baseInitializer.CalculateSize() != sameGraphInitializer.CalculateSize()) continue;
			if (!AreDimsEqual(baseInitializer.Dims, sameGraphInitializer.Dims)) continue;

			var data1 = baseInitializer.RawData.AsSpan();
			var data2 = sameGraphInitializer.RawData;
			if (data1.SequenceEqual(data2)) {
				yield return (sameGraphInitializer.Name, baseInitializer.CalculateSize());
			}
		}
	}

	private static ModelProto? LoadModel(String modelName) {
		// Span<Byte> modelData = File.ReadAllBytes(Path.ChangeExtension(modelName, "onnx")).AsSpan();

		ModelProto? model = Serializer.Deserialize(File.OpenRead(modelName), default(ModelProto));
		if (model == null) {
			Console.Error.WriteLine($"Failed to load '{modelName}'");
			return null;
		}

		Console.WriteLine($"Successfully loaded '{modelName}' ({new FileInfo(modelName).Length.ToFileSize()}) for ONNX 1.{model.IrVersion}-{(Version)model.IrVersion} OpSet:{String.Join(", ", model.OpsetImports.Select(ops => ops.Domain == String.Empty ? $"ONNX:{ops.Version}" : $"{ops.Domain}:{ops.Version}"))} with ModelVersion:{model.ModelVersion} by {model.ProducerName}@{model.ProducerVersion}");
		return model;
	}

	private static void SaveModel(String filename, ModelProto model) {
		using FileStream fs = new(filename, FileMode.Create, FileAccess.Write, FileShare.None);
		Serializer.Serialize(fs, model);
	}

	private static Boolean IsDataEqual(TensorProto initializer1, TensorProto initializer2) {
		ReadOnlySpan<Byte> rawData1 = initializer1.GetRawData();
		ReadOnlySpan<Byte> rawData2 = initializer2.GetRawData();

		return rawData1.SequenceEqual(rawData2);
	}

	private static Boolean AreDimsEqual(Int64[]? baseInitializer, Int64[]? sameGraphInitializer) {
		if (baseInitializer == null && sameGraphInitializer == null) return true;
		if ((baseInitializer == null && sameGraphInitializer != null) || (baseInitializer != null && sameGraphInitializer == null)) return false;
		if (baseInitializer != null && sameGraphInitializer != null) return baseInitializer.SequenceEqual(sameGraphInitializer);

		return false;
	}

	public static void Merge(String encoderFilename, String decoderFilename, String target) {
		ModelProto model1 = LoadModel(encoderFilename) ?? throw new ArgumentException($"Unable to load onnx model from {encoderFilename}", nameof(encoderFilename));
		ModelProto model2 = LoadModel(decoderFilename) ?? throw new ArgumentException($"Unable to load onnx model from {decoderFilename}", nameof(decoderFilename));

		Dictionary<String, String> nodeInOutNameOverrides = new();
		nodeInOutNameOverrides.Add("encoder_attention_mask", "attention_mask");
		nodeInOutNameOverrides.Add("encoder_hidden_states", "last_hidden_state");
		// nodeInOutNameOverrides.Add("input_ids", "input_ids");

		RenameNode(model2, "input_ids", "decoder_input_ids");
		model1.Graph.Inputs.Add(model2.Graph.Inputs.First(n => n.Name == "decoder_input_ids"));

		model1.Graph.Outputs.Clear();
		model1.Graph.Outputs.AddRange(model2.Graph.Outputs);
		var originalNodeInOutNames = model1.Graph.Nodes.SelectMany(node => node.Inputs).Concat(model1.Graph.Nodes.SelectMany(node => node.Outputs)).ToFrozenSet();
		var originalInitializerNames = model1.Graph.Initializers.Select(ini => ini.Name).ToFrozenSet();
		List<NodeProto> oldNodes = model1.Graph.Nodes.ToList();
		foreach (NodeProto model2Node in model2.Graph.Nodes) {
			if (oldNodes.Exists(n => n.Name == model2Node.Name)) {
				Console.WriteLine($"Conflict for node name {model2Node.Name}");
				model2Node.Name = "dec____" + model2Node.Name;
			}

			for (int index = 0; index < model2Node.Inputs.Count; index++) {
				string model2NodeInput = model2Node.Inputs[index];
				if (originalNodeInOutNames.Contains(model2NodeInput)) {
					Boolean isInitializerConflict = originalInitializerNames.Contains(model2NodeInput);
					Boolean alreadyRenamed = nodeInOutNameOverrides.ContainsKey(model2NodeInput);
					if (!isInitializerConflict && !alreadyRenamed)
						nodeInOutNameOverrides.Add(model2NodeInput, "dec____" + model2NodeInput);
					Console.WriteLine($"Conflict for node {model2Node.Name}, duplicate input name #{index + 1}/{model2Node.Inputs.Count} = {model2NodeInput} -- {(isInitializerConflict ? "Initializer" : (alreadyRenamed ? "alreadyRenamed" : "Conflict"))}");
				}
			}

			for (int index = 0; index < model2Node.Outputs.Count; index++) {
				string model2NodeOutput = model2Node.Outputs[index];
				if (originalNodeInOutNames.Contains(model2NodeOutput)) {
					Boolean alreadyRenamed = nodeInOutNameOverrides.ContainsKey(model2NodeOutput);
					if (!alreadyRenamed)
						nodeInOutNameOverrides.Add(model2NodeOutput, "dec____" + model2NodeOutput);
					Console.WriteLine($"Conflict for node {model2Node.Name}, duplicate output name #{index + 1}/{model2Node.Outputs.Count} = {model2NodeOutput} -- {(alreadyRenamed ? "alreadyRenamed" : "Conflict")}");
				}
			}

			foreach (KeyValuePair<String, String> nodeInOutNameOverride in nodeInOutNameOverrides) {
				Int32 idx = model2Node.Inputs.IndexOf(nodeInOutNameOverride.Key);
				if (idx != -1) model2Node.Inputs[idx] = nodeInOutNameOverride.Value;
				idx = model2Node.Outputs.IndexOf(nodeInOutNameOverride.Key);
				if (idx != -1) model2Node.Outputs[idx] = nodeInOutNameOverride.Value;
			}

			model1.Graph.Nodes.Add(model2Node);
		}

		foreach (var valInfo in model2.Graph.ValueInfoes) {
			foreach (KeyValuePair<String, String> nodeInOutNameOverride in nodeInOutNameOverrides) {
				if (valInfo.Name.Equals(nodeInOutNameOverride.Key))
					valInfo.Name = nodeInOutNameOverride.Value;
			}

			model1.Graph.ValueInfoes.Add(valInfo);
		}

		foreach (TensorProto initializerToAdd in model2.Graph.Initializers) {
			TensorProto? existingInitializerData = model1.Graph.Initializers.Find(i => AreDimsEqual(initializerToAdd.Dims, i.Dims) && IsDataEqual(i, initializerToAdd));
			if (existingInitializerData == null) {
				model1.Graph.Initializers.Add(initializerToAdd);
			} else {
				foreach (NodeProto graphNode in model1.Graph.Nodes) {
					Int32 idx = graphNode.Inputs.IndexOf(initializerToAdd.Name);
					if (idx == -1) continue;

					// Console.WriteLine($"Changing input {graphNode.Name}[{idx}] from {graphNode.Inputs[idx]}({String.Join("x", initializerToAdd.Dims ?? [])}) to existing initializer {existingInitializerData.Name}({String.Join("x", existingInitializerData.Dims ?? [])})");
					graphNode.Inputs[idx] = existingInitializerData.Name;
				}
			}
		}

		SaveModel(target, model1);
	}

	private static void RenameNode(ModelProto model, String originalName, String targetName) {
		foreach (ValueInfoProto node in model.Graph.Inputs.Concat(model.Graph.Outputs).Concat(model.Graph.ValueInfoes).Where(n => n.Name == originalName)) {
			node.Name = targetName;
		}

		foreach (NodeProto node in model.Graph.Nodes) {
			if(node.Name == originalName) node.Name = targetName;
			Int32 idx = node.Inputs.IndexOf(originalName);
			if (idx >= 0) {
				node.Inputs.RemoveAt(idx);
				node.Inputs.Insert(idx, targetName);
			}
			idx = node.Outputs.IndexOf(originalName);
			if (idx >= 0) {
				node.Outputs.RemoveAt(idx);
				node.Outputs.Insert(idx, targetName);
			}
		}
		
		foreach (TensorProto node in model.Graph.Initializers.Where(n => n.Name == originalName)) {
			node.Name = targetName;
		}

		if (model.Graph.QuantizationAnnotations.Any() || model.Graph.SparseInitializers.Any())
			throw new InvalidOperationException("Graphs with QuantizationAnnotations or SparseInitializers not supported.");

	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="filename"></param>
	/// <param name="smallestSizeToSplit">Inclusive size in bytes. Default 1 KiB, so everything 1 KiB and larger is extracted</param>
	/// <exception cref="ArgumentException"></exception>
	public static void Split(String filename, Int32 smallestSizeToSplit = 1 * 1024*1024) {
		ExtractInitializers(filename, smallestSizeToSplit);
		return;

		static void ExtractInitializers(String filename, Int32 smallestSizeToSplit) {
			ModelProto model = LoadModel(filename) ?? throw new ArgumentException($"Unable to load onnx model from {filename}", nameof(filename));

			String initializerTargetDirectory = Path.Combine(Path.GetDirectoryName(filename) ?? ".", "initializers");
			Directory.CreateDirectory(initializerTargetDirectory);

			IEnumerable<TensorProto> initializers = model.Graph.Initializers;
			Dictionary<String, InitializerDefinition> initializerDefinitions = new();
			foreach (TensorProto initializer in initializers) {
				ReadOnlySpan<Byte> rawData = initializer.GetRawData();
				if (rawData.Length < smallestSizeToSplit) continue;
				String initalizerDataHash = XxHash3.HashToUInt64(rawData).ToString("X");
				String initializerFilename = Path.Combine(initializerTargetDirectory, initalizerDataHash);
				if (!File.Exists(initializerFilename)) {
					File.WriteAllBytes(initializerFilename, rawData);
				}

				initializerDefinitions.Add(initializer.Name, new InitializerDefinition(initalizerDataHash, ((TensorProto.DataType)initializer.data_type).ToString(), initializer.Dims));
				TensorShapeProto shape = new();
				shape.Dims.AddRange(initializer.Dims.Select(dim => new TensorShapeProto.Dimension { DimValue = dim }));
				
				model.Graph.Inputs.Add(new ValueInfoProto() {
					Name = initializer.Name,
					Type = new TypeProto() {
						TensorType = new TypeProto.Tensor() {
							ElemType = initializer.data_type,
							Shape = shape,
						},
					},
				});

				initializer.RawData = null;
				initializer.Int32Datas = null;
				initializer.Int64Datas = null;
				initializer.Uint64Datas = null;
				initializer.DoubleDatas = null;
				initializer.FloatDatas = null;

				// TODO add to input so they become overridable
				// TODO model1.Graph.ValueInfoes;
				// TODO model1.Graph.Nodes;
			}

			model.Graph.Initializers.RemoveAll(initializer => initializerDefinitions.ContainsKey(initializer.Name));
			// model.Graph.Inputs.AddRange(model.Graph.ValueInfoes.Where(vi => initializerDefinitions.ContainsKey(vi.Name)));
			// model.Graph.ValueInfoes.RemoveAll(vi => initializerDefinitions.ContainsKey(vi.Name));

			// model1.Graph.Initializers.Clear();
			SaveModel(Path.ChangeExtension(filename, ".clear.onnx"), model);
			using var fs1 = File.OpenWrite(Path.ChangeExtension(filename, "clear.json"));
			JsonSerializer.Serialize(fs1, initializerDefinitions);
		}
	}
}