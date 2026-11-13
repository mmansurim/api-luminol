import os
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

# Diretório para salvar temporariamente as imagens recebidas e processadas
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def processar_luminol(imagem_path):
  # 1. Carregar a imagem
  img = cv2.imread(imagem_path)
  if img is None:
    return None, 'Erro ao carregar a imagem'

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 2. Detectar a placa de Petri (HoughCircles) para calibração exata (aprox. 9 cm)
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

  PIXELS_POR_CM2 = 100.0  # Fallback padrão caso não detecte o círculo
  raio_petri_cm = 4.5
  circulo_detectado = False

  if circles is not None:
    circles = np.uint16(np.around(circles))
    c = circles[0, 0]
    centro_x, centro_y, raio_pixels = c[0], c[1], c[2]
    pixels_por_cm = raio_pixels / raio_petri_cm
    PIXELS_POR_CM2 = pixels_por_cm**2
    circulo_detectado = True

  # 3. Processamento HSV para identificar o brilho e a reação de luminol
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv)

  _, thresh_brilho = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)

  lower_reacao = np.array([0, 50, 150])
  upper_reacao = np.array([180, 255, 255])
  mask_cor = cv2.inRange(hsv, lower_reacao, upper_reacao)

  mask_final = cv2.bitwise_and(thresh_brilho, mask_cor)

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
    if area_cnt > 50:
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
    cv2.circle(resultado, (centro_x, centro_y), raio_pixels, (255, 0, 0), 2)

  # Salvar imagem processada
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
  if 'image' not in request.files:
    return jsonify({'erro': 'Nenhuma imagem enviada'}), 400

  file = request.files['image']
  filepath = os.path.join(UPLOAD_FOLDER, 'recebido.jpg')
  file.save(filepath)

  dados, erro = processar_luminol(filepath)
  if erro:
    return jsonify({'erro': erro}), 500

  return jsonify(dados)


@app.route('/resultado/<filename>')
def obter_resultado(filename):
  return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/')
def home():
  return jsonify({
      'status': 'online',
      'mensagem': 'API de Detecção e Quantificação de Luminol está ativa!',
  })
  
if __name__ == '__main__':
  # Porta dinâmica exigida por plataformas de nuvem (como o Render)
  port = int(os.environ.get('PORT', 5000))
  app.run(host='0.0.0.0', port=port)
