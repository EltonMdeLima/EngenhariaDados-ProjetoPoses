import cv2
import mediapipe as mp
import pandas as pd
import sqlite3
import os
import glob
import json
import logging
from datetime import datetime

# --- 0. Configuração e Logging ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Definição dos caminhos DENTRO do container (mapeados no docker-compose)
VIDEO_INPUT_DIR = "/app/videos_input"
KEYPOINTS_OUTPUT_DIR = "/app/keypoints_output"
DATABASE_DIR = "/app/database"
DB_FILE = os.path.join(DATABASE_DIR, "poses.db")

# Inicializa as ferramentas do MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, 
                    model_complexity=1, # 0 (rápido), 1 (médio), 2 (lento/preciso)
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def setup_database(db_path):
    """Cria a tabela no SQLite se ela não existir."""
    logging.info(f"Configurando banco de dados em: {db_path}")
    os.makedirs(os.path.dirname(db_path), exist_ok=True) # Garante que a pasta /database exista
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tabela principal para armazenar todos os keypoints
    # Usamos uma chave composta (video_nome, frame) para evitar duplicatas
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS keypoints_normalizados (
        video_nome TEXT,
        frame INTEGER,
        landmark_id INTEGER,
        landmark_nome TEXT,
        x REAL,
        y REAL,
        z REAL,
        visibilidade REAL,
        data_processamento TEXT,
        PRIMARY KEY (video_nome, frame, landmark_id)
    )
    """)
    conn.commit()
    conn.close()
    logging.info("Banco de dados pronto.")

def extrair_keypoints(video_path, output_dir):
    """
    (E) Extrai - Processa um vídeo e salva os keypoints como JSONs.
    Retorna o nome base do vídeo para referência.
    """
    video_nome = os.path.basename(video_path)
    logging.info(f"[E]xtraindo keypoints do vídeo: {video_nome}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Não foi possível abrir o vídeo: {video_path}")
        return None

    frame_count = 0
    dados_brutos = [] # Armazena todos os keypoints deste vídeo
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break # Fim do vídeo

        # Converte a imagem (BGR para RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Processa a imagem com o MediaPipe
        results = pose.process(image_rgb)
        
        # Coleta os keypoints se detectados
        if results.pose_landmarks:
            frame_data = {'frame': frame_count, 'landmarks': []}
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_data = {
                    'id': idx,
                    'nome': mp_pose.PoseLandmark(idx).name,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
                frame_data['landmarks'].append(landmark_data)
            dados_brutos.append(frame_data)
        
        frame_count += 1

    cap.release()

    # Salva os dados brutos (JSON) - Boa prática de Data Lake
    if dados_brutos:
        json_output_path = os.path.join(output_dir, f"{video_nome}.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(json_output_path, 'w') as f:
            json.dump(dados_brutos, f, indent=2)
        logging.info(f"[E]xtração concluída. {frame_count} frames processados.")
        return video_nome, json_output_path
        
    logging.warning(f"Nenhuma pose detectada em: {video_nome}")
    return None, None

def transformar_e_carregar(video_nome, json_path, db_path):
    """
    (T+L) Transforma os JSONs e Carrega no SQLite.
    """
    if not json_path:
        return

    logging.info(f"[T+L] Processando {video_nome} para o banco de dados...")
    
    # (T) Transformação: Lê o JSON e o "achata" (flatten)
    with open(json_path, 'r') as f:
        dados_brutos = json.load(f)

    linhas_para_db = []
    data_proc_str = datetime.utcnow().isoformat()

    for frame_data in dados_brutos:
        frame_id = frame_data['frame']
        for landmark in frame_data['landmarks']:
            linhas_para_db.append((
                video_nome,
                frame_id,
                landmark['id'],
                landmark['nome'],
                landmark['x'],
                landmark['y'],
                landmark['z'],
                landmark['visibility'],
                data_proc_str
            ))

    if not linhas_para_db:
        logging.warning("Nenhuma linha para carregar após transformação.")
        return

    # (L) Carga: Conecta ao DB e insere
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Usamos 'INSERT OR IGNORE' para evitar erros de duplicatas
        query = """
        INSERT OR IGNORE INTO keypoints_normalizados 
        (video_nome, frame, landmark_id, landmark_nome, x, y, z, visibilidade, data_processamento)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor.executemany(query, linhas_para_db)
        conn.commit()
        
        logging.info(f"[T+L] Sucesso. {cursor.rowcount} novos keypoints inseridos no banco.")
        
    except sqlite3.Error as e:
        logging.error(f"Erro ao carregar dados no SQLite: {e}")
    finally:
        if conn:
            conn.close()

def main():
    """Função principal que orquestra o pipeline."""
    logging.info("--- INICIANDO PIPELINE DE PROCESSAMENTO DE POSES ---")
    
    # 0. Configura o Banco de Dados
    setup_database(DB_FILE)
    
    # 1. Busca por vídeos na pasta de entrada
    videos = glob.glob(os.path.join(VIDEO_INPUT_DIR, "*.mp4"))
    if not videos:
        logging.warning(f"Nenhum vídeo .mp4 encontrado em {VIDEO_INPUT_DIR}")
        return

    logging.info(f"Encontrados {len(videos)} vídeos para processar.")

    # 2. Roda o pipeline (ETL) para cada vídeo
    for video_path in videos:
        try:
            # (E) Extrair
            video_nome, json_path = extrair_keypoints(video_path, KEYPOINTS_OUTPUT_DIR)
            
            # (T+L) Transformar e Carregar
            if video_nome:
                transformar_e_carregar(video_nome, json_path, DB_FILE)
                
        except Exception as e:
            logging.error(f"Falha ao processar o vídeo {video_path}: {e}")

    logging.info("--- PIPELINE CONCLUÍDO ---")

if __name__ == "__main__":
    main()