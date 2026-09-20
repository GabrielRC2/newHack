# Frequência por visão computacional — MVP

Backend FastAPI para uma única câmera USB na porta. Não usa direção, tracking, linha virtual ou múltiplas câmeras: para cada aluno e aula, um reconhecimento válido abre um intervalo (entrada); o próximo o fecha (saída); o seguinte abre retorno. Um `cooldown` de 15 segundos protege contra frames repetidos.

## Requisitos e execução

Python 3.10 ou superior é necessário.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Abra `http://127.0.0.1:8000/docs` para a documentação interativa. Em outro terminal, a interface opcional:

```powershell
streamlit run streamlit_app.py
```

O SQLite é criado como `frequency.db`. Para configurar a instalação, copie `.env.example` e defina as variáveis de ambiente no terminal.

## Horários oficiais

O endpoint `POST /sessions/scheduled` cria uma aula a partir destes índices:

| índice | horário |
|---|---|
| 1 | 07:55–08:50 |
| 2 | 08:50–10:10 |
| 3 | 10:10–12:00 |
| 4 | 13:30–15:20 |
| 5 | 15:20–17:35 |
| 6 | 19:00–21:50 |

Exemplo:

```json
POST /sessions/scheduled
{"class_date":"2026-09-20","schedule_index":5,"label":"Turma A","minimum_percentage":0.75}
```

## Fluxo da API

1. Cadastre o aluno com o primeiro embedding:

```json
POST /students
{"name":"Ana Silva","enrollment_number":"2026001","embedding":[0.12,0.98,0.44]}
```

2. O módulo de visão envia a detecção. `class_id` é o identificador da `ClassSession` retornado ao criar a aula:

```json
POST /recognitions
{"class_id":1,"camera_id":"usb-entrada-01","embedding":[0.12,0.98,0.44],"timestamp":"2026-09-20T15:00:00"}
```

3. Consulte `GET /sessions/1/attendance` ou `GET /sessions/1/students/1/attendance`.

4. Ao acabar, execute `POST /sessions/1/finalize`. O endpoint fecha intervalos ainda abertos em `ends_at`, soma somente as interseções com o horário oficial e marca `present` quando o total é maior ou igual a `minimum_percentage` (75% por padrão).

No exemplo 15:00–15:10 e 16:00–16:30, o total é **40 minutos**. Conforme a regra solicitada, um novo cadastro inicia com `Student.em_aula=true`; o primeiro reconhecimento cria o intervalo auditável que será contabilizado. Depois, `em_aula` fica em `true` num intervalo aberto e `Student.tempo_em_aula` é atualizado em segundos; o resumo confiável por aula é `Attendance.total_seconds`.

## Endpoints

- `POST /students` — aluno e primeiro embedding
- `POST /students/{student_id}/embeddings` — embedding adicional
- `POST /sessions` — aula com horários manuais
- `POST /sessions/scheduled` — aula pelos horários oficiais
- `POST /recognitions` — evento do módulo de visão
- `GET /sessions/{class_id}/attendance` — presença da aula
- `GET /sessions/{class_id}/students/{student_id}/attendance` — resumo individual
- `POST /sessions/{class_id}/finalize` — fecha e calcula presença


## Reconhecimento facial

O backend não recebe imagens: o cliente local em `vision_client/` detecta o rosto, gera um embedding com YuNet + SFace do OpenCV e envia somente o vetor numérico para `POST /recognitions`. OpenCV possui suporte publicado para Python 3.14, e as APIs `FaceDetectorYN`/`FaceRecognizerSF` usadas pelo cliente são as recomendadas na documentação oficial do OpenCV. [OpenCV PyPI](https://pypi.org/project/opencv-python/) e [tutorial do modelo](https://docs.opencv.org/5.0/tutorials/dnn/dnn_face/dnn_face.html).

Com a API já iniciada, instale as dependências adicionais e baixe os dois modelos ONNX uma vez:

```bash
python -m pip install -r requirements.txt
python -m vision_client.download_models
```

Cadastre o aluno usando o mesmo modelo da câmera. Na janela, mantenha apenas um rosto e pressione `C` três vezes, variando levemente a pose; `Q` cancela sem enviar dados.

```bash
python -m vision_client.enroll --name "Ana Silva" --enrollment-number "2026001"
```

Crie uma aula e anote o ID retornado. Em seguida, execute a câmera:

```bash
python -m vision_client.run_camera --class-id 1 --camera-id usb-entrada-01
```

O cliente envia **um evento por passagem** e só rearma após um segundo sem rosto no enquadramento. Isso impede que a pessoa parada diante da câmera gere uma falsa saída após o cooldown. Quando há mais de um rosto, ele não envia nada e pede que passe um aluno por vez. Altere `--camera-index 1` se a webcam USB não for o índice `0`.

O valor inicial de `FACE_MATCH_THRESHOLD` foi ajustado para `0.45`, adequado para embeddings SFace normalizados como ponto de partida. Ele não é uma garantia: valide o limiar com dados consentidos da instituição antes de operação real.

## Limites, privacidade e segurança

- Câmera única não permite saber a direção física: por contrato, o estado atual é a fonte da entrada/saída. Uma “saída sem entrada” não pode ser distinguida de uma entrada, portanto inicia um intervalo e fica registrada como `entered`.
- Obtenha consentimento explícito, forneça alternativa manual e jamais use biometria como único mecanismo de punição/decisão contestável.
- Embeddings são dados biométricos sensíveis: use HTTPS, controle de acesso, criptografia em repouso/chaves fora do código, auditoria e retenção mínima. Este MVP persiste embeddings em JSON apenas para demonstração; em produção, proteja-os em armazenamento apropriado e descarte eventos/imagens brutas.
- Ajuste o limiar com dados autorizados e monitore falsos positivos/negativos antes de uso real.
- Este cliente não implementa prova de vida/anti-spoofing. Não trate uma foto/tela como impossível; para produção, acrescente liveness detection e revisão humana para casos contestados.
