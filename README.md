# Limiarização Adaptativa com OpenCV: Segmentando Imagens com Inteligência Local

Gabriella Braz - 10402554  
Giovana Liao - 10402264  
Maria Julia de Pádua - 10400630

## 1. Introdução
A limiarização é uma técnica de processamento de imagens que transforma imagens em tons de cinza em preto e branco, facilitando a identificação de objetos e contornos. Isso é feito comparando o brilho de cada pixel com um valor limite (limiar): pixels mais claros viram branco e os mais escuros, preto. Essa abordagem ajuda tanto na visualização quanto na análise por algoritmos de visão computacional.
Entretanto, em imagens com variações de iluminação, como sombras ou reflexos, a limiarização tradicional pode apresentar problemas. Isso ocorre porque ela utiliza um único valor de limiar para a imagem inteira. Quando diferentes regiões da imagem apresentam níveis distintos de iluminação, esse único valor pode causar perda de detalhes em áreas muito claras ou muito escuras.

Para contornar essa limitação, a limiarização adaptativa é empregada. Ao invés de aplicar um valor único, essa técnica divide a imagem em pequenas regiões e calcula um limiar local para cada uma delas, considerando a média ou a média ponderada dos pixels vizinhos. Dessa forma, mesmo com variações de luz em diferentes partes da imagem, é possível preservar melhor os detalhes e obter um resultado mais equilibrado.
Este artigo apresenta exemplos práticos de aplicação da limiarização adaptativa utilizando a linguagem Python e a biblioteca OpenCV. É mostrado o passo a passo do processo, desde o carregamento da imagem até a aplicação dos métodos adaptativos baseados na média e na média gaussiana. Os resultados são comparados visualmente, permitindo analisar as diferenças e compreender as situações em que cada método pode ser mais eficiente.

## 2. Conceitos fundamentais
A limiarização simples consiste em definir um valor de corte: todos os pixels acima desse valor se tornam brancos, e os abaixo, pretos. Essa técnica funciona bem quando a iluminação é uniforme.
Problema com a Limiarização Global
Imagine uma foto tirada ao ar livre com áreas iluminadas pelo sol e outras à sombra. Um único valor de limiar pode eliminar detalhes importantes de uma das regiões, já que a percepção de brilho varia em diferentes partes da imagem.

**Limiarização Adaptativa**

A limiarização adaptativa resolve esse problema ao aplicar limiares locais em pequenas janelas da imagem. Existem duas principais abordagens:
Média local: o limiar é a média dos pixels vizinhos.
Média gaussiana: semelhante à média local, mas os pixels mais próximos ao centro da região têm mais peso.
Essas técnicas proporcionam uma segmentação mais equilibrada e eficiente em imagens com variações de iluminação.

## 3. Implementação dos métodos em Python

**Pré-requisitos**

```python
pip install opencv-python matplotlib numpy pillow
```

**Carregamento da Imagem e Pré-Processamento**

O código começa com a importação de algumas bibliotecas importantes. Essas bibliotecas ajudam o programa a trabalhar com imagens, números e gráficos:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib
```

O código acessa uma imagem da internet, lê os dados em bytes e os transforma em um formato que o OpenCV consegue entender. 

```python
# 1. Baixa a imagem da internet
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Taylor_Swift_at_the_Golden_Globes_2024_%28Enhanced%2C_cropped%29_2.jpg/640px-Taylor_Swift_at_the_Golden_Globes_2024_%28Enhanced%2C_cropped%29_2.jpg'
resposta = urllib.request.urlopen(url)
imagem_bytes = bytearray(resposta.read())
imagem_array = np.asarray(imagem_bytes, dtype=np.uint8)
imagem = cv2.imdecode(imagem_array, cv2.IMREAD_COLOR)
```

Como o objetivo é trabalhar com a luz da imagem (e não com as cores), o programa transforma a imagem colorida em uma versão em preto e branco (tons de cinza):

```python
# 2. Converte para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
```

O código suaviza a imagem usando um tipo de desfoque chamado “Gaussiano”. Isso ajuda a reduzir ruídos e melhora o resultado da limiarização:

```python
# 3. Aplica desfoque gaussiano
imagem_borrada = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)

# 4. Limiarização simples (global)
_, limiar_global = cv2.threshold(imagem_borrada, 127, 255, cv2.THRESH_BINARY_INV)
```

**Método 1: Limiarização Adaptativa com Média**

Esse método divide a imagem em pequenas áreas, em vez de analisar a imagem inteira de uma vez só. Dentro de cada uma dessas regiões menores, ele calcula a média dos tons de cinza. Depois, compara o valor de cada pixel com essa média local. Se o valor do pixel for menor que a média, ele vira branco (ou preto, dependendo da configuração); se for maior, vira o oposto. Isso ajuda a destacar detalhes mesmo quando a iluminação da imagem varia bastante de um ponto para outro.

```python
# 5. Limiarização adaptativa - MÉDIA
thresh_media = cv2.adaptiveThreshold(imagem_borrada, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
```

**Método 2: Limiarização Adaptativa com Gaussiana**

Assim como a limiarização pela média, esse método também divide a imagem em várias regiões menores para analisar os pixels localmente. No entanto, ao invés de calcular uma média simples dentro dessas regiões, ele usa uma média ponderada, em que os pixels mais próximos do centro da área analisada têm mais peso no cálculo do limiar. Esse tipo de média é feita com a ajuda de uma função chamada Gaussiana, que suaviza os valores. O resultado costuma ser mais suave e mais eficiente em imagens com variações de iluminação, pois ajuda a destacar melhor os contornos e detalhes, reduzindo os ruídos que poderiam atrapalhar a segmentação.

```python
# 6. Limiarização adaptativa - GAUSSIANA
thresh_gauss = cv2.adaptiveThreshold(imagem_borrada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
```

## 4. Resultados e discussão
Por fim, o código exibe as imagens usando o Matplotlib.
Ele mostra a versão em tons de cinza, a limiarização por média, lado a lado, para que seja possível comparar os resultados.

```python
# 7. Mostra as imagens lado a lado
fig, axs = plt.subplots(1, 5, figsize=(22, 5))


axs[0].imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
axs[0].set_title("Imagem Original")
axs[0].axis("off")


axs[1].imshow(imagem_cinza, cmap='gray')
axs[1].set_title("Imagem em Tons de Cinza")
axs[1].axis("off")


axs[2].imshow(limiar_global, cmap='gray')
axs[2].set_title("Limiarização Global")
axs[2].axis("off")


axs[3].imshow(thresh_media, cmap='gray')
axs[3].set_title("Adaptativa - Média")
axs[3].axis("off")


axs[4].imshow(thresh_gauss, cmap='gray')
axs[4].set_title("Adaptativa - Gaussiana")
axs[4].axis("off")


plt.tight_layout()
plt.show()
```

**Análise**
Média simples: boa para destacar regiões bem delimitadas, mas pode ser mais sensível a ruídos.
Gaussiana: mais suave, preserva contornos com melhor fidelidade mesmo em áreas com transições suaves de iluminação.

**Possíveis melhorias:**
Ajustar o tamanho da janela e o valor da constante (C) para refinar os resultados.
Testar em diferentes tipos de imagem (documentos, rostos, ambientes externos).
Combinar com técnicas morfológicas para pós-processamento (remoção de ruídos).


## 5. Conclusão
A limiarização adaptativa se mostrou uma técnica eficiente para lidar com imagens que apresentam variações de iluminação, superando as limitações da limiarização tradicional com valor fixo. Ao aplicar métodos como o da média e o gaussiano, conseguimos destacar melhor os elementos da imagem mesmo em condições não uniformes de luz. Essa abordagem é especialmente útil em aplicações de visão computacional, como reconhecimento de padrões e segmentação de objetos, onde a qualidade da imagem pode variar bastante. O uso de bibliotecas como OpenCV, NumPy e Matplotlib facilita todo o processo de carregamento, processamento e visualização das imagens.

## 6. Bibliografia
VALLIM, Gabriel. Segmentação de imagens utilizando técnicas de thresholding. Medium, 26 abr. 2019. Disponível em: https://medium.com/data-hackers/segmenta%C3%A7%C3%A3o-de-imagens-utilizando-t%C3%A9cnicas-dethresholding-1ee031562c63. Acesso em: 6 abr. 2025.
AEROENGENHARIA. O que é limiarização em processamento de imagem? Disponível em: https://aeroengenharia.com/glossario/o-que-e-limiarizacao-em-processamento-de-imagem/. Acesso em: 6 abr. 2025.
OPENCV. Image Thresholding (Python). Disponível em: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html. Acesso em: 6 abr. 2025.
FISHER, Robert B. Adaptive Thresholding. University of Edinburgh. Disponível em: https://homepages.inf.ed.ac.uk/rbf/HIPR2/adpthrsh.htm. Acesso em: 6 abr. 2025.
CODER, Pulo do Gato. Thresholding no OpenCV (Limiarização). YouTube, 24 jan. 2021. Disponível em: https://www.youtube.com/watch?v=65Xbn1IuYqk. Acesso em: 6 abr. 2025.
CODER, Pulo do Gato. Thresholding adaptativo no OpenCV (limiarização adaptativa). YouTube, 25 jan. 2021. Disponível em: https://www.youtube.com/watch?v=1lkOTltVsQ8. Acesso em: 6 abr. 2025.
