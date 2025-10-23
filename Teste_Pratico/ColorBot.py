# Importa as bibliotecas necessárias
import cv2
import numpy as np
import argparse
import time
import os

# Define a função de segmentação por HSV
def segment_hsv(image, target_color, hsv_overrides):
    # Converte a imagem BGR para HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define os dicionários de ranges de cor padrão
    default_ranges = {
        'green': ([30, 100, 50], [89, 255, 255]),
        'blue': ([90, 100, 50], [135, 255, 255])
    }

    # Obtém os ranges padrão para a cor alvo
    lower_default, upper_default = default_ranges[target_color]

    # Cria o array numpy para o limite inferior, usando overrides se existirem
    lower_bound = np.array([
        hsv_overrides.get('hmin', lower_default[0]),
        hsv_overrides.get('smin', lower_default[1]),
        hsv_overrides.get('vmin', lower_default[2])
    ])

    # Cria o array numpy para o limite superior, usando overrides se existirem
    upper_bound = np.array([
        hsv_overrides.get('hmax', upper_default[0]),
        hsv_overrides.get('smax', upper_default[1]),
        hsv_overrides.get('vmax', upper_default[2])
    ])

    # Cria a máscara binária usando os limites
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # Aplica erosão para remover ruído
    mask = cv2.erode(mask, None, iterations=1)
    # Aplica dilatação para restaurar o tamanho do objeto
    mask = cv2.dilate(mask, None, iterations=1)
    # Retorna a máscara processada
    return mask

def segment_kmeans(image, target_color, k):
    # Converte a imagem BGR para HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Reformata a imagem para uma lista de pixels
    pixel_data = hsv_image.reshape((-1, 3))
    # Converte os dados para float32, necessário para o K-Means
    pixel_data = np.float32(pixel_data)

    # Define os critérios de parada do K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Executa o algoritmo K-Means
    ret, labels, centers = cv2.kmeans(pixel_data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Define a faixa de Hue para verificação
    default_ranges = {
        'green': ([30], [90]), # H_min=30, H_max=90
        'blue': ([90], [135])  # H_min=90, H_max=130
    }

    # Extrai os limites de Matiz (H)
    h_min = np.array(default_ranges[target_color][0])
    h_max = np.array(default_ranges[target_color][1])

    # Isso evita que clusters de cinza/preto/branco sejam selecionados
    s_min_threshold = 130
    v_min_threshold = 50

    matching_cluster_indices = []

    for i, center in enumerate(centers):
        h_centroid, s_centroid, v_centroid = center[0], center[1], center[2]

        # Verifica se o Hue está na faixa
        is_in_hue_range = (h_centroid >= h_min and h_centroid <= h_max)

        # Verifica se a cor não é cinza/preta
        is_a_color = (s_centroid >= s_min_threshold and v_centroid >= v_min_threshold)

        if is_in_hue_range and is_a_color:
            # Adiciona o índice do cluster
            matching_cluster_indices.append(i)

    final_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Se encontramos clusters correspondentes
    if len(matching_cluster_indices) > 0:
        # Combina as máscaras de todos os clusters que correspondem
        for idx in matching_cluster_indices:
            cluster_mask = (labels == idx).reshape(image.shape[:2])
            final_mask = cv2.bitwise_or(final_mask, np.uint8(cluster_mask * 255))


    # Se nenhum cluster se encaixar, usa distancia da centroide para escolher um
    else:
        # Usando as cores padrão
        target_hsv_colors = {
            'green': np.array([60, 180, 130], dtype=np.float32),
            'blue':  np.array([120, 180, 130], dtype=np.float32)
        }
        target_hsv = target_hsv_colors[target_color]

        def calculate_weighted_hsv_distance(center, target):
            h_center, s_center, v_center = center
            h_target, s_target, v_target = target

            # Calcula a distância do Hue em um círculo (0-179)
            diff_h = abs(h_center - h_target)
            dist_h = min(diff_h, 180 - diff_h) # Distância circular

            # Calcula a distância de Saturação e Luz
            dist_s = abs(s_center - s_target)
            dist_v = abs(v_center - v_target)

            # Retorna uma distância ponderada:
            return (dist_h * 3.0) + (dist_s * 1.0) + (dist_v * 1.0)

        # Calcula as distâncias
        distances = [calculate_weighted_hsv_distance(center, target_hsv) for center in centers]
        # Pega o cluster que possui menor distância ponderada
        target_cluster = np.argmin(distances)

        # Cria a máscara selecionando pixels que pertencem a esse único cluster
        mask_array = (labels == target_cluster).reshape(image.shape[:2])
        final_mask = np.uint8(mask_array * 255)

    return final_mask

# Define a função que cria o overlay
def create_overlay(image, mask, target_color):
    # Converte a máscara de 1 canal (escala de cinza) para 3 canais (BGR)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Mistura a imagem original (30%) com a máscara BGR (70%)
    overlay = cv2.addWeighted(image, 0.3, mask_bgr, 0.7, 0)
    # Retorna a imagem de overlay
    return overlay

# Define a função principal do programa
def main():
    # Inicializa o 'parser' de argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Teste Técnico - Segmentação de Imagem")

    # Adiciona o argumento para o caminho da imagem
    parser.add_argument('--input', type=str, required=False, help="Caminho para a imagem de entrada.")
    # Adiciona o argumento para usar a webcam
    parser.add_argument('--webcam', action='store_true', help="Usar a webcam como entrada.")

    # Adiciona o argumento para o método de segmentação (obrigatório)
    parser.add_argument('--method', type=str, required=True, choices=['hsv', 'kmeans'], help="Método de segmentação.")
    # Adiciona o argumento para a cor alvo (obrigatório)
    parser.add_argument('--target', type=str, required=True, choices=['green', 'blue'], help="Cor alvo para segmentação.")

    # Adiciona o argumento para o número de clusters (padrão 3)
    parser.add_argument('--k', type=int, default=3, help="Número de clusters (K) para o K-Means.")

    # Adiciona argumentos para sobrepor os valores HSV
    parser.add_argument('--hmin', type=int, help="Override: HUE mínimo")
    parser.add_argument('--hmax', type=int, help="Override: HUE máximo")
    parser.add_argument('--smin', type=int, help="Override: Saturação mínima")
    parser.add_argument('--smax', type=int, help="Override: Saturação máxima")
    parser.add_argument('--vmin', type=int, help="Override: Valor (Brilho) mínimo")
    parser.add_argument('--vmax', type=int, help="Override: Valor (Brilho) máximo")

    # Analisa os argumentos fornecidos na linha de comando
    args = parser.parse_args()

    # Valida se o usuário forneceu uma e apenas uma fonte de entrada
    if not args.webcam and args.input is None:
        parser.error("Você deve fornecer --input ou --webcam.")
    if args.webcam and args.input is not None:
        parser.error("--input e --webcam não podem ser usados juntos.")

    # Cria um dicionário com os overrides de HSV
    hsv_overrides = {
        'hmin': args.hmin, 'hmax': args.hmax,
        'smin': args.smin, 'smax': args.smax,
        'vmin': args.vmin, 'vmax': args.vmax,
    }
    # Filtra o dicionário, mantendo apenas os overrides que foram fornecidos
    hsv_overrides = {k: v for k, v in hsv_overrides.items() if v is not None}

    # Verifica se o modo webcam foi ativado
    if args.webcam:
        # Inicia o modo webcam
        print("Iniciando modo webcam... Pressione 'q' para sair.")
        # Inicializa a captura de vídeo
        cap = cv2.VideoCapture(0)

        # Verifica se a webcam foi aberta com sucesso
        if not cap.isOpened():
            print("Erro: Não foi possível abrir a webcam.")
            return

        # Inicia o loop de processamento em tempo real
        while True:
            # Marca o tempo de início do frame
            start_time = time.time()
            # Lê um frame da webcam
            ret, frame = cap.read()
            # Se a leitura falhar, encerra o loop
            if not ret:
                print("Erro: Não foi possível ler o frame.")
                break

            # Inverte o frame horizontalmente (efeito espelho)
            frame = cv2.flip(frame, 1)

            # Inicializa a máscara como Nula
            mask = None
            # Se o método for HSV
            if args.method == 'hsv':
                # Processa o frame completo com HSV
                mask = segment_hsv(frame, args.target, hsv_overrides)
            # Se o método for K-Means
            elif args.method == 'kmeans':
                # Obtém as dimensões do frame
                h, w = frame.shape[:2]
                # Define uma escala para reduzir o frame
                scale = 400 / w
                # Redimensiona o frame para uma versão menor
                small_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

                # Executa o K-Means no frame pequeno
                small_mask = segment_kmeans(small_frame, args.target, args.k)

                # Redimensiona a máscara pequena de volta ao tamanho original do frame
                mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Cria a imagem de overlay
            overlay_image = create_overlay(frame, mask, args.target)


            # Marca o tempo de fim do frame
            end_time = time.time()
            # Calcula o tempo de execução do frame
            exec_time = end_time - start_time
            # Calcula o FPS (Frames Por Segundo)
            fps = 1 / max(exec_time, 1e-6)

            # Calcula o número total de pixels
            total_pixels = mask.size
            # Conta quantos pixels foram segmentados
            segmented_pixels = cv2.countNonZero(mask)
            # Calcula a porcentagem de pixels segmentados
            percentage = (segmented_pixels / total_pixels) * 100

            # Imprime o status no terminal
            print(f"FPS: {fps:.2f} | Segmentado: {percentage:.2f}%", end='\r')


            # Mostra a janela de overlay
            cv2.imshow("ColorBot - Overlay", overlay_image)
            # Mostra a janela da máscara
            cv2.imshow("ColorBot - Mask", mask)

            # Verifica se a tecla 'q' foi pressionada para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        # Libera o dispositivo de captura
        cap.release()
        # Fecha todas as janelas abertas pelo OpenCV
        cv2.destroyAllWindows()
        # Imprime uma nova linha após o log de FPS
        print("\nWebcam desligada.")

    # Se não for modo webcam, é modo arquivo
    else:
        # Inicia o modo arquivo
        print(f"Processando arquivo: {args.input}")
        # Marca o tempo de início do processamento
        start_time = time.time()

        # Carrega a imagem do arquivo
        image = cv2.imread(args.input)
        # Verifica se a imagem foi carregada
        if image is None:
            print(f"Erro: Não foi possível carregar a imagem em {args.input}")
            return

        # Inicializa a máscara como Nula
        mask = None

        # Se o método for HSV
        if args.method == 'hsv':
            # Processa a imagem com HSV
            mask = segment_hsv(image, args.target, hsv_overrides)
        # Se o método for K-Means
        elif args.method == 'kmeans':
            # Avisa o usuário que pode demorar
            print("Processando K-Means, isso pode demorar um pouco...")
            # Obtem as dimensões da imagem
            h, w = image.shape[:2]
            # Define uma escala para reduzir a imagem
            scale = 800 / w
            # Redimensiona a imagem para uma versão menor
            small_frame = cv2.resize(image, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            # Executa o K-Means na imagem pequena
            small_mask = segment_kmeans(small_frame, args.target, args.k)

            # Redimensiona a máscara pequena de volta ao tamanho original da imagem
            mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)


        # Verifica se a máscara foi criada com sucesso
        if mask is None:
            print("Erro: A máscara não pôde ser gerada.")
            return


        # Define o nome da pasta de saída
        output_dir = "outputs"
        # Cria a pasta 'outputs' se ela não existir
        os.makedirs(output_dir, exist_ok=True)
        # Obtém o nome base do arquivo de entrada (sem extensão)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        # Cria um sufixo para o nome do arquivo baseado no método
        method_name = f"{args.method}_{args.target}" + (f"_k{args.k}" if args.method == 'kmeans' else "")
        # Define o caminho completo para salvar a máscara
        mask_filename = os.path.join(output_dir, f"{base_name}_{method_name}_mask.png")
        # Define o caminho completo para salvar o overlay
        overlay_filename = os.path.join(output_dir, f"{base_name}_{method_name}_overlay.png")

        # Cria a imagem de overlay
        overlay_image = create_overlay(image, mask, args.target)

        # Salva o arquivo da máscara
        cv2.imwrite(mask_filename, mask)
        # Salva o arquivo de overlay
        cv2.imwrite(overlay_filename, overlay_image)


        # Marca o tempo de fim do processamento
        end_time = time.time()
        # Calcula o tempo total de execução
        exec_time = end_time - start_time
        # Calcula o total de pixels
        total_pixels = mask.size
        # Conta os pixels segmentados
        segmented_pixels = cv2.countNonZero(mask)
        # Calcula a porcentagem segmentada
        percentage = (segmented_pixels / total_pixels) * 100

        # Imprime o resumo do processamento
        print("--- Processamento Concluído ---")
        print(f"Método: {args.method} (Alvo: {args.target})")
        print(f"Tempo de execução: {exec_time:.4f} segundos")
        print(f"Pixels segmentados: {segmented_pixels} de {total_pixels} ({percentage:.2f}%)")
        print(f"Máscara salva em: {mask_filename}")
        print(f"Overlay salvo em: {overlay_filename}")


if __name__ == "__main__":
    main()