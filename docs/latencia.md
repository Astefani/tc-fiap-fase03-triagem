# Latência: modelo original × otimizado

O modelo servido é o PubMedBERT (com fine tuning), exportado para ONNX e quantizado para INT8.  
Este documento mostra quanto essa otimização ganha em latência, a que custo de qualidade e como foi medido.  
Os números foram gerados por [`scripts/bench_latency.py`](../scripts/bench_latency.py) e estão disponíveis nos arquivos JSON em [`latencia/`](latencia/).

## Resumo

- **INT8 é 2,7× mais rápido que o modelo original** nos três percentis avaliados (p50, p95 e p99).
- **O ganho vem da quantização:** só exportar para ONNX (fp32) ganha 8%; do fp32 para o INT8, 2,5×.
- **Custo de qualidade dentro do ruído:** F1-macro 0,6918 × 0,6950 do original (−0,3 pt).
- **Tamanho do modelo:** 4× menor (105 × 418 MB).
- **A API custa ~1 ms:** p50 de 34,9 ms via HTTP × 33,7 ms chamando o modelo direto.

## Resultados


| Backend | p50 | p95 | p99 | Média | Requisições/s¹ | Tamanho | F1-macro |
|---|---|---|---|---|---|---|---|
| PyTorch (original, sem otimização) | 90,6 ms | 103,9 ms | 104,1 ms | 83,0 ms |  12,0 | 418 MB | 0,6950 |
| ONNX fp32 | 83,8 ms | 97,1 ms | 97,2 ms | 76,0 ms | 13,2 | 418 MB | 0,6967 |
| **ONNX INT8 (servido)** | **33,7 ms** | **39,1 ms** | **39,2 ms** | **30,6 ms** | **32,7** | **105 MB** | **0,6918** |
| TF-IDF + LogReg (referência) | 0,54 ms | 0,63 ms | 0,67 ms | 0,54 ms | 1.842 | 4 MB | 0,6649 |

¹ Uma requisição por vez (sem concorrência): 1.000 / latência média.  

**Pela API (HTTP)**, com o mesmo protocolo e o modelo num container preso aos mesmos núcleos:

| Backend | p50 | p95 | p99 | Média | Requisições/s | Custo da API sobre o modelo (p50) |
|---|---|---|---|---|---|---|
| ONNX fp32 | 86,3 ms | 98,8 ms | 106,1 ms | 78,1 ms | 12,8 | +2,5 ms |
| **ONNX INT8** | **34,9 ms** | **40,4 ms** | **40,9 ms** | **31,8 ms** | **31,4** | **+1,2 ms** |
| TF-IDF + LogReg | 1,35 ms | 4,35 ms | 5,60 ms | 1,80 ms | 556,8 | +0,8 ms |

## Protocolo de medição

- **Entrada:** os 947 resumos únicos da validação, na ordem em que aparecem no arquivo de treino.
- **Medição:** a lista é percorrida em ciclo até 1.050 chamadas sequenciais de um texto cada: as 50 primeiras são
  aquecimento (descartadas) e as 1.000 seguintes são medidas.
- **Dois níveis:** *in-process* chama o modelo direto (tokenização + inferência); *HTTP* chama a API no container
  (soma validação, serialização, framework e rede local).
- **Mesmas condições para todos:**
  - 4 threads de inferência (ONNX Runtime e PyTorch);
  - modelo preso a 4 núcleos de performance físicos;
  - no modo HTTP, o cliente roda em outros núcleos de performance físicos para não disputar CPU com a API;
  - máquina sem outra carga.
- **Um backend por vez**, com pausa entre eles (ver a lição abaixo).
- **Rastreável:** cada JSON registra CPU, núcleos usados, versões das bibliotecas e o commit do código.

| | |
|---|---|
| CPU | Intel Core i5-12600K (6 núcleos de performance + 4 de eficiência, x86_64) |
| Sistema | Ubuntu 26.4.1, Python 3.12, onnxruntime e tokenizers nas versões do `poetry.lock` |
| Código | commit `ee349c6` (in-process) e `448aab9` (HTTP; mesmo código de inferência) |


## Limitações

- Os números absolutos valem para esta CPU.
- Teste realizados com processador M4 apresentaram resultados diferentes, evidenciando que o conjunto infraestrutura/sistema pode impactar a performance.
- Uma rodada por backend; a variação entre rodadas observada no projeto é de 5–10%.
- O F1 de cada variante é o da validação; a avaliação no teste oficial está no README.