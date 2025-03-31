# IM_OCDM_ISAC
Modificação do sinal Index OCDM para ISAC comm + sensing
# IM_OCDM_ISAC
Doc Inicial
---

# ISAC System with IM-OCDM and LFM Radar

## Visão Geral do Projeto
Este projeto implementa um sistema integrado de comunicação e sensoriamento (ISAC) que combina:
- **Comunicação**: Modulação por índice (IM) em Orthogonal Chirp Division Multiplexing (OCDM)
- **Sensoriamento**: Radar com modulação linear de frequência (LFM)

**Objetivo principal**: Compartilhar eficientemente o espectro entre transmissão de dados e operações de radar.

### Características-Chave
- Uso de 256 subchirps divididos igualmente (128 para comunicação, 128 para radar)
- Eficiência espectral através de ativação esparsa (25% dos subchirps para comunicação)
- Implementação de radar pulsado com sinal LFM
- Modulação 16-QAM para comunicação digital

---

## Funções Principais

### 1. `index_mapper(index_bits, n_c, k, g)`-->--> segue o paper "2024_Orthogonal_Chirp_Division_Multiplexing_With_Index_Modulation"
**Propósito**: Mapeia bits de índice para subchirps ativos  
**Entradas**:
- `index_bits`: Vetor binário com seleções de subchirps
- `n_c`: Subchirps por bloco
- `k`: Subchirps ativos por bloco
- `g`: Número de blocos

**Saída**:  
`active_subchirps` - Índices dos subchirps ativos (1×64)

**Funcionamento**:  
1. Divide os bits de índice em grupos por bloco
2. Converte cada grupo para decimal
3. Seleciona combinações de subchirps usando tabela nchoosek
4. Converte para índices globais

### 2. `generate_lfm_chirp(BW, pulse_width, Fs, N)`
**Propósito**: Gera sinal radar LFM  
**Entradas**:
- `BW`: Largura de banda do chirp
- `pulse_width`: Duração do pulso
- `Fs`: Frequência de amostragem
- `N`: Número de amostras

**Saída**:  
`lfm_signal` - Sinal LFM no domínio do tempo

**Funcionamento**:
- Cria sinal com variação linear de frequência
- Formato: `f(t) = -BW/2 → +BW/2`

### 3. `process_radar(radar_signal, lfm_template, Fs, pulse_width)`
**Propósito**: Processa eco radar para estimar alcance e velocidade  (ainda em ajuste) 
**Entradas**:
- `radar_signal`: Sinal recebido
- `lfm_template`: Sinal LFM de referência
- `Fs`: Frequência de amostragem
- `pulse_width`: Duração do pulso

**Saídas**:
- `range_est`: Alcance estimado (metros)
- `velocity_est`: Velocidade estimada (m/s)

**Funcionamento**:
1. Filtro casado para detecção de alcance
2. Detecção de pico para estimativa de tempo de voo
3. Cálculo de velocidade (implementação futura com FFT Doppler)

---

## Fluxo Principal do Sistema

### 1. Configuração
```matlab
Nc_total = 256;    % Total de subchirps
Nc_comm = 128;     % Subchirps para comunicação
g = 8;             % Blocos de comunicação
k = 8;             % Subchirps ativos por bloco
```

### 2. Geração de Sinais
- **Comunicação**:
  - Ativa 64 subchirps via index modulation
  - Modulação 16-QAM para transmissão de dados

- **Radar**:
  - Ocupa 192 subchirps inativos
  - Sinal LFM com BW = 1 MHz

### 3. Processamento
- **Receptor de Comunicação**:
  - Demodulação 16-QAM
  - Reconstrução de bits

- **Receptor Radar**:
  - Correlação com template LFM
  - Estimativa de alcance

---

## Instruções de Uso (GitHub)

### Pré-requisitos
- MATLAB R2020b ou superior
- Signal Processing Toolbox

### Execução
1. Clone o repositório
2. Execute `basic_imocdm.m`
3. Resultados serão exibidos em:
   - Gráfico de constelação 16-QAM
   - Sinal radar no domínio do tempo
   - Estimativas de alcance/velocidade no console

---

## Contribuições e Melhorias Futuras

### Potenciais Aprimoramentos
1. Implementação completa do processamento Doppler
2.  Implementação da Rede Neural LSTM para classificar Comm de Radar (em andamento)
3. Adição de canal de multipercurso realista
4. Técnicas de redução de PAPR
5. Suporte para múltiplos usuários



### Contribuição
1. Faça um fork do repositório
2. Crie sua branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Add feature'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

---

Este documento está disponível em [README.md](https://github.com/seu-usuario/ISAC-IM-OCDM/blob/main/README.md) no repositório principal.
