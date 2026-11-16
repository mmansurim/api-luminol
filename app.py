import base64
import os
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def processar_luminol(imagem_path):
  img = cv2.imread(imagem_path)
  if img is None:
    return None, 'Erro ao carregar a imagem'

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 1. Detectar a placa de Petri (HoughCircles) para calibração e limite geográfico
  circles = cv2.HoughCircles(
      cv2.medianBlur(gray, 5),
      cv2.HOUGH_GRADIENT,
      dp=1.2,
      minDist=100,
      param1=50,
      param2=30,
      minRadius=100,
      maxRadius=400,
  )

  PIXELS_POR_CM2 = 100.0  # Fallback padrão
  raio_petri_cm = 4.5
  circulo_detectado = False
  centro_x, centro_y, raio_pixels = 0, 0, 0

  if circles is not None:
    circles = np.uint16(np.around(circles))
    c = circles[0, 0]
    centro_x, centro_y, raio_pixels = c[0], c[1], c[2]
    pixels_por_cm = raio_pixels / raio_petri_cm
    PIXELS_POR_CM2 = pixels_por_cm**2
    circulo_detectado = True

  # 2. Processamento HSV focado na cor da mancha de reação (tons avermelhados/oxidados)
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  lower_red1 = np.array([0, 50, 50])
  upper_red1 = np.array([15, 255, 255])
  lower_red2 = np.array([165, 50, 50])
  upper_red2 = np.array([180, 255, 255])

  mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
  mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
  mask_cor = cv2.bitwise_or(mask1, mask2)

  # 3. Restrição Geográfica: Isolar a busca estritamente DENTRO da placa de Petri detectada
  if circulo_detectado:
    mask_petri = np.zeros_like(mask_cor)
    # Criamos um círculo branco interno para mascarar apenas a área útil da placa
    cv2.circle(
        mask_petri, (centro_x, centro_y), raio_pixels - 5, 255, -1
    )
    mask_final = cv2.bitwise_and(mask_cor, mask_petri)
  else:
    mask_final = mask_cor  # Se não achar o círculo, analisa a imagem inteira

  # Operações morfológicas para limpeza de ruídos
  kernel = np.ones((5, 5), np.uint8)
  mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel)
  mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel)

  contours, _ = cv2.findContours(
      mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  resultado = img.copy()
  deteccao_encontrada = False
  area_total_pixels = 0

  for cnt in contours:
    area_cnt = cv2.contourArea(cnt)
    # Filtro para ignorar pequenas imperfeições ou reflexos menores que 80 pixels
    if area_cnt > 80:
      deteccao_encontrada = True
      area_total_pixels += area_cnt

      x, y, w, h = cv2.boundingRect(cnt)
      cv2.rectangle(resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)

      area_cm2 = area_cnt / PIXELS_POR_CM2
      texto = f'Luminol: {area_cm2:.2f} cm2'
      cv2.putText(
          resultado,
          texto,
          (x, y - 10),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 0),
          2,
      )

  if circulo_detectado:
    cv2.circle(
        resultado, (centro_x, centro_y), raio_pixels, (255, 0, 0), 2
    )  # Borda azul da placa

  output_filename = 'resultado_processado.jpg'
  output_path = os.path.join(UPLOAD_FOLDER, output_filename)
  cv2.imwrite(output_path, resultado)

  area_total_cm2 = (
      (area_total_pixels / PIXELS_POR_CM2) if deteccao_encontrada else 0.0
  )

  return {
      'detectado': deteccao_encontrada,
      'area_pixels': int(area_total_pixels),
      'area_cm2': round(float(area_total_cm2), 2),
      'imagem_url': f'/resultado/{output_filename}',
  }, None


@app.route('/analisar', methods=['POST'])
def analisar():
  print('--- NOVA REQUISIÇÃO RECEBIDA DO APP ---')
  print('FILES:', request.files)
  print('FORM:', request.form)

  filepath = os.path.join(UPLOAD_FOLDER, 'recebido.jpg')
  ficheiro_salvo = False

  # 1. Tentar apanhar se veio através de request.files (qualquer chave multipart)
  if request.files:
    for chave in request.files:
      file = request.files[chave]
      if file and file.filename != '':
        file.save(filepath)
        ficheiro_salvo = True
        print(f"Ficheiro guardado com sucesso via request.files['{chave}']")
        break

  # 2. Tentar apanhar se veio como dados binários crus (request.data)
  if not ficheiro_salvo and request.data:
    with open(filepath, 'wb') as f:
      f.write(request.data)
    ficheiro_salvo = True
    print('Ficheiro guardado com sucesso via request.data binário')

  # 3. Tentar ler diretamente do stream de rede se ainda não foi salvo
  if not ficheiro_salvo:
    try:
      dados_stream = request.stream.read()
      if dados_stream and len(dados_stream) > 0:
        with open(filepath, 'wb') as f:
          f.write(dados_stream)
        ficheiro_salvo = True
        print('Ficheiro guardado com sucesso via request.stream')
    except Exception as e:
      print('Erro ao ler stream:', str(e))

  # Se mesmo assim nenhum método conseguiu extrair o ficheiro
  if not ficheiro_salvo or not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
    print('ERRO: Nenhum conteúdo de imagem válido encontrado.')
    return (
        jsonify({
            'erro': 'Nenhuma imagem enviada (corpo da requisição vazio)',
            'detectado': False,
            'imagem_url': '',
        }),
        400,
    )

  # Processar a imagem com o OpenCV
  dados, erro = processar_luminol(filepath)
  if erro:
    return jsonify({'erro': erro, 'detectado': False, 'imagem_url': ''}), 500

  return jsonify(dados)

@app.route('/analisarAPP', methods=['POST'])
def analisar_app():
    filepath = os.path.join(UPLOAD_FOLDER, 'luminol_recebida.jpg')
    ficheiro_salvo = False

    try:
        print(f"Content-Type recebido: {request.content_type}")
        print(f"Chaves em request.files: {list(request.files.keys())}")

        # 1. O Web1.PostarFicheiro envia via multipart/form-data (verificar PRIMEIRO)
        if request.files:
            for chave in request.files:
                file = request.files[chave]
                if file and file.filename != '':
                    file.save(filepath)
                    ficheiro_salvo = True
                    print(f"Ficheiro guardado com sucesso via request.files['{chave}']")
                    break

        # 2. Alternativa secundária caso venha por bytes brutos
        if not ficheiro_salvo and request.data and len(request.data) > 0:
            with open(filepath, "wb") as f:
                f.write(request.data)
            ficheiro_salvo = True
            print("Ficheiro guardado via request.data")

        # Validação se o ficheiro chegou e tem tamanho real
        if not ficheiro_salvo or not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return jsonify({
                "status": "erro",
                "mensagem": "Nenhuma imagem válida foi enviada no corpo da requisição."
            }), 400

        # 3. Processamento OpenCV (exemplo)
        imagem = cv2.imread(filepath)
        if imagem is None:
            return jsonify({
                "status": "erro",
                "mensagem": "O servidor não conseguiu descodificar a imagem."
            }), 500

        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        pixels_brilhantes = int(np.sum(thresh == 255))
        area_cm2 = round(pixels_brilhantes * 0.001, 2)

        return jsonify({
            "status": "sucesso",
            "mensagem": "Análise efetuada com sucesso",
            "area_cm2": area_cm2,
            "pixels_detectados": pixels_brilhantes
        }), 200

    except Exception as e:
        print(f"Erro interno: {str(e)}")
        return jsonify({
            "status": "erro",
            "mensagem": f"Erro interno: {str(e)}"
        }), 500

@app.route('/resultado/<filename>')
def obter_resultado(filename):
  return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/')
def home():
  return jsonify({
      'status': 'online',
      'mensagem': 'API de Detecção e Quantificação de Luminol ativa!',
  })


if __name__ == '__main__':
  port = int(os.environ.get('PORT', 5000))
  app.run(host='0.0.0.0', port=port)
