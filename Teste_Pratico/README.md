# Teste Técnico - Segmentação de Imagem (YBYSYS)

Este projeto é um aplicativo de linha de comando (CLI) em Python para segmentar imagens com base na cor, utilizando dois métodos diferentes: Ranges HSV e Agrupamento K-Means.

## Instalação

1. Esse aplicativo foi projetado em Python 3.9.13, portanto uma versão mais recente ou equivalente é necessária. Para descobrir a versão instalada abra seu terminal e digite:
   ```bash
   python --version
   ```
2. Abra seu terminal e navegue até o local da pasta:

    Caso sua pasta esteja no desktop, será necessário antes da execução digitar:
    ```bash
    cd desktop/Teste_Pratico
    ```
    Caso sua pasta esteja em outro local, será necessário especificar o caminho desta depois do comando `cd`. Com isso, agora é possível executar o aplicativo e usar as pastas disponíveis neste local;
3. Por fim, instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
   Com o caminho da pasta principal especificado no terminal, basta rodar o comando acima para instalar as bibliotecas necessárias para o uso do aplicativo.

## Como Rodar

Esse aplicativo roda a partir do terminal ao executar o script `ColorBot.py`. 

Uma pasta chamada `samples/` está disponível com imagens de teste. Para usar as imagens é necessário especificar o caminho delas como nos exemplos abaixo.

No caso de carregar arquivos de imagem, o script criará uma pasta `outputs/` para salvar os resultados, dentro da pasta principal do aplicativo.
Se o método escolhido for a WebCam, os resultados serão apresentados ao vivo.

Com o caminho especificado, digite `python ColorBot.py --help` para verificar os comandos disponíveis. 


Abaixo deixo exemplos de uso para facilitar a execução:

**Método HSV:**

1. Verde padrão, usando os valores definidos no código

```bash
python ColorBot.py --input samples/wally.jpg --method hsv --target green
```
2. Verde com alteração no Hue
```bash
python ColorBot.py --webcam --method hsv --target green --hmin 30 --hmax 90
```
3. Verde com alteração no valor mínimo da saturação
```bash
python ColorBot.py --input samples/wally.jpg --method hsv --target blue --smin 100
```
4. Verde com alteração no valor mínimo da luz
```bash
python ColorBot.py --input samples/wally.jpg --method hsv --target blue --vmin 100
```
5. Verde com alteração nos valores máximos de saturação e de luz
```bash
python ColorBot.py --input samples/wally.jpg --method hsv --target green --smax 150 --vmax 150
```


**Método K-Means:**
6. K-Means com K=2
```bash
python ColorBot.py --webcam --method kmeans --target green --k 2
```

7. K-Means com K=3
```bash
python ColorBot.py --input samples/wally.jpg --method kmeans --target blue --k 3
```
8. K-Means com K=4
```bash
python ColorBot.py --input samples/wally.jpg --method kmeans --target green --k 4
```
9. K-Means com K=10
```bash
python ColorBot.py --webcam --method kmeans --target blue --k 10
```

## Explicação dos Métodos:
**Segmentação por Cor (HSV):**

Este método converte a imagem para o espaço de cor HSV (Hue, Saturation, Value). O HSV é útil porque separa a cor (Hue) da luz (Value) e saturação, tornando-o mais eficiente na segmentação por cores do que o RGB. O aplicativo aplica um "filtro" que seleciona apenas os pixels dentro de um intervalo de H, S e V definido para a cor-alvo. 

Os pixels selecionados são então tratados como valor máximo, dentro de uma imagem onde todo o resto é preto, para criar a máscara principal. Após a criação da máscara, a imagem principal e a própria máscara são mescladas para formação de uma imagem de overlay.

**Segmentação por Agrupamento (K-Means):**

Este é um algoritmo de machine learning não supervisionado. Ele agrupa todos os pixels da imagem em 'K' grupos distintos, chamados de clusters. A ideia é que pixels de cores semelhantes (ex: tons de verde) acabarão no mesmo cluster. O aplicativo então analisa a cor média (centróide) de cada cluster e seleciona o cluster cuja cor média é mais próxima da cor-alvo (verde ou azul puros).

O aplicativo aplica o algoritmo K-Means na imagem recebida, verifica quais dos clusters obtidos esta dentro dos parâmetros da cor alvo e depois os junta na máscara. Caso nenhum cluster satisfaça os requisitos, através da distancia da cor da centróide do cluster com o alvo, é selecionado o único cluster que mais se aproxima da cor do alvo, gerando a máscara e posteriormente o overlay.


## Escolha dos ranges:
Para escolha dos ranges no método HSV foram utilizados como padrão os valores que abrangem a maior parte dos tons de verdes e azuis, podendo existir variações quanto ao reconhecimento a depender da luminosidade do local, já que a alteração dos parâmetros do HSV pode ser feita pelo usuário para restringir os tons quando necessário.

Para o método K-Means, o espaço de cores HSV também foi selecionado para esse método mediante testes realizados com ele e com o espaço de cores RGB. Durante os testes, foi verificado menor tempo de processamento ao utilizar o espaço de cores HSV em comparação com o RGB. Além do Hue para verificar a cor alvo, foram utilizados valores de saturação e luz mínimos para evitar cinzas e pretos.


## Limitações verificadas:
Durante os testes, pode-se perceber que durante a execução do método HSV, quando um brilho muito forte, como o da tela do celular em nível máximo é apontado para a câmera, mesmo que a cor azul esteja preenchendo toda a tela o que acaba sendo marcado pelo aplicativo é a luz emitida e não a cor em si. Esse efeito também ocorre sob luz solar intensa.

Já para o método K-Means, o método de selecionar o cluster mais próximo da cor alvo, não foi muito eficaz e resultou em muitos erros por conta da diferença entre tons da mesma cor. Por conta disso, a verificação dos parâmetros HSV foi necessária para filtras os clusters que se encaixavam com a cor alvo e obter melhor precisão. Também foi verificado que em imagens com muitas cores diferentes como o caso do wally.jpg, o algoritmo tende a dar muitos falsos positivos para a cor alvo. Portanto em casos como esses, um número alto de clusters deve ser utilizado para isolar melhor a cor alvo, o que pode tornar a resposta um pouco lenta. 

Quando nenhum cluster se encaixa nos parâmetros HSV, a cor alvo não existe na imagem, e então a distancia da centróide do cluster que mais se aproxima do alvo é utilizada para selecionar o cluster mais próximo do alvo, como requerido.

