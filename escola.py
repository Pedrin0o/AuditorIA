import os
import json
import time
from dotenv import load_dotenv
import ollama
from groq import Groq
from openai import OpenAI, RateLimitError

# Carrega as chaves do arquivo .env
load_dotenv()

# --- Clientes das 2 provedoras "professoras" ---
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# OpenRouter usa o SDK da OpenAI, só trocando a base_url
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# --- Nomes dos modelos (atualizados em julho/2026) ---
MODELO_ALUNO = "llama3.2:1b"
MODELO_PROFESSOR_1 = "openai/gpt-oss-20b"          # Groq - avaliação rápida
MODELO_PROFESSOR_2 = "openai/gpt-oss-120b"         # Groq - lição de ouro final
MODELO_PROFESSOR_3 = "meta-llama/llama-3.3-70b-instruct:free"  # OpenRouter - segunda opinião

# Limite de segurança: quantas rodadas de "dúvidas" no máximo por tema.
# Isso evita loop infinito, já que o modelo local (1B) nem sempre para sozinho.
MAX_RODADAS_DUVIDA = 5

PALAVRA_PARADA = "SEM_DUVIDAS"


def aluno_responde_inicial(tema):
    """O Aluno local tenta responder o tema pela primeira vez."""
    print("\n[Estudante Local] Pensando na primeira resposta...")
    try:
        response = ollama.chat(model=MODELO_ALUNO, messages=[
            {'role': 'system', 'content': 'Você é um estudante iniciante e prestativo. Responda de forma simples.'},
            {'role': 'user', 'content': tema}
        ])
        return response['message']['content']
    except Exception as e:
        return f"[Erro no Aluno]: {e}"


def aluno_pergunta_duvida(tema, historico_texto):
    """
    O Aluno lê tudo que já foi discutido e decide:
    - Fazer UMA pergunta específica sobre algo que não ficou claro, OU
    - Responder exatamente a palavra de parada, se não tiver mais dúvidas.
    """
    print("\n[Estudante Local] Pensando se tem alguma dúvida...")
    prompt = f"""Você é um estudante estudando o tema "{tema}".
Abaixo está tudo que já foi ensinado a você sobre esse tema até agora:

{historico_texto}

Se ainda tiver alguma dúvida específica sobre algo que não ficou claro ou que você quer aprofundar,
faça UMA pergunta curta e direta sobre isso.

Se você já entendeu tudo sobre esse tema e não tem mais nenhuma dúvida, responda APENAS com a palavra:
{PALAVRA_PARADA}
"""
    try:
        response = ollama.chat(model=MODELO_ALUNO, messages=[
            {'role': 'system', 'content': 'Você é um estudante curioso. Siga exatamente o formato pedido.'},
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content'].strip()
    except Exception as e:
        print(f"[Erro no Aluno ao pensar em dúvida]: {e}")
        return PALAVRA_PARADA  # Se der erro, encerra a conversa por segurança


def professor_1_avalia(pergunta, resposta_aluno):
    """Professor 1 (Groq - gpt-oss-20b): avaliação rápida e nota da tentativa inicial do aluno."""
    print("\n[Professor 1 - Groq/gpt-oss-20b] Avaliando resposta do aluno...")
    prompt = f"""Como professor de IA, avalie a resposta que o aluno deu para a pergunta abaixo.
Aponte o que está certo, o que está errado (inclusive alucinações/dados inventados) e dê uma nota de 0 a 10.

Pergunta: {pergunta}
Resposta do Aluno: {resposta_aluno}"""
    try:
        completion = groq_client.chat.completions.create(
            model=MODELO_PROFESSOR_1,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Erro na API da Groq (Professor 1): {e}")
        return "Feedback indisponível devido a erro."


def professor_1_responde_duvida(duvida, historico_texto):
    """Professor 1 (Groq): dá uma primeira resposta rápida para a dúvida do aluno."""
    print("\n[Professor 1 - Groq/gpt-oss-20b] Respondendo à dúvida...")
    prompt = f"""Contexto do que já foi ensinado:
{historico_texto}

O aluno tem a seguinte dúvida: {duvida}

Responda a dúvida de forma clara, correta e objetiva."""
    try:
        completion = groq_client.chat.completions.create(
            model=MODELO_PROFESSOR_1,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Erro na API da Groq (Professor 1): {e}")
        return "Resposta indisponível devido a erro."


def professor_3_segunda_opiniao(pergunta, resposta_aluno):
    """Professor 3 (OpenRouter - Llama 3.3 70B): segunda opinião sobre a tentativa inicial do aluno."""
    print("\n[Professor 3 - OpenRouter/Llama 3.3 70B] Dando uma segunda opinião...")
    prompt = f"""Você é um segundo avaliador independente. Avalie criticamente a resposta do aluno abaixo,
apontando erros factuais, imprecisões ou informações inventadas. Seja direto.

Pergunta: {pergunta}
Resposta do Aluno: {resposta_aluno}"""
    return _chamar_openrouter(prompt)


def professor_3_responde_duvida(duvida, historico_texto):
    """Professor 3 (OpenRouter): segunda opinião sobre a resposta à dúvida do aluno."""
    print("\n[Professor 3 - OpenRouter/Llama 3.3 70B] Dando uma segunda opinião sobre a dúvida...")
    prompt = f"""Contexto do que já foi ensinado:
{historico_texto}

O aluno tem a seguinte dúvida: {duvida}

Dê sua própria resposta a essa dúvida, de forma independente e objetiva."""
    return _chamar_openrouter(prompt)


def _chamar_openrouter(prompt):
    """Função auxiliar: chama o OpenRouter com retry automático em caso de rate limit (429)."""
    try:
        completion = openrouter_client.chat.completions.create(
            model=MODELO_PROFESSOR_3,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except RateLimitError:
        espera = 30
        print(f"Professor 3 está sobrecarregado (rate limit). Aguardando {espera}s para tentar de novo...")
        time.sleep(espera)
        try:
            completion = openrouter_client.chat.completions.create(
                model=MODELO_PROFESSOR_3,
                messages=[{"role": "user", "content": prompt}]
            )
            return completion.choices[0].message.content
        except Exception as e2:
            print(f"Erro na API do OpenRouter (Professor 3) na segunda tentativa: {e2}")
            return "Segunda opinião indisponível devido a erro (rate limit persistente)."
    except Exception as e:
        print(f"Erro na API do OpenRouter (Professor 3): {e}")
        return "Segunda opinião indisponível devido a erro."


def professor_2_ensina(pergunta, resposta_aluno, feedback_1, feedback_3):
    """Professor 2 (Groq - gpt-oss-120b): junta as opiniões e escreve a lição perfeita inicial."""
    print("\n[Professor 2 - Groq/gpt-oss-120b] Gerando a lição de ouro...")
    prompt = f"""Você é o Diretor da Escola de IA. Com base na pergunta, na tentativa do aluno e no feedback
de dois outros professores, escreva a resposta didática perfeita que o aluno deveria ter dado.
Seja claro, direto e corrija qualquer erro factual mencionado pelos professores.

Pergunta original: {pergunta}
Resposta do aluno: {resposta_aluno}
Feedback do Professor 1: {feedback_1}
Feedback do Professor 3: {feedback_3}"""
    try:
        completion = groq_client.chat.completions.create(
            model=MODELO_PROFESSOR_2,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Erro na API da Groq (Professor 2): {e}")
        return "Lição indisponível devido a erro."


def professor_2_responde_duvida(duvida, historico_texto, feedback_1, feedback_3):
    """Professor 2 (Groq): junta as duas respostas e escreve a resposta final para a dúvida."""
    print("\n[Professor 2 - Groq/gpt-oss-120b] Gerando a resposta final para a dúvida...")
    prompt = f"""Você é o Diretor da Escola de IA. Um aluno teve uma dúvida durante uma aula. Com base no
contexto da aula, na dúvida do aluno e nas respostas de dois outros professores, escreva a resposta
final, clara e correta, que resolve a dúvida do aluno.

Contexto da aula até agora:
{historico_texto}

Dúvida do aluno: {duvida}
Resposta do Professor 1: {feedback_1}
Resposta do Professor 3: {feedback_3}"""
    try:
        completion = groq_client.chat.completions.create(
            model=MODELO_PROFESSOR_2,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Erro na API da Groq (Professor 2): {e}")
        return "Lição indisponível devido a erro."


def salvar_conversa_treino(tema, conversa):
    """
    Salva a conversa completa (tema + todas as perguntas/lições) como UM exemplo
    multi-turno de treino, no formato padrão de fine-tuning.
    """
    mensagens = [{"role": "user", "content": tema}]
    for turno in conversa:
        mensagens.append({"role": "assistant", "content": turno["licao"]})
        if turno.get("proxima_pergunta"):
            mensagens.append({"role": "user", "content": turno["proxima_pergunta"]})

    dados = {"messages": mensagens}
    with open("dados_de_treino.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dados, ensure_ascii=False) + "\n")
    print(f"\n[Sistema] Conversa completa sobre '{tema}' salva em 'dados_de_treino.jsonl' ({len(conversa)} rodada(s))!")


def montar_historico_texto(tema, conversa):
    """Monta um texto legível com tudo que já foi discutido, para dar contexto às IAs."""
    partes = [f"Tema: {tema}"]
    for i, turno in enumerate(conversa, start=1):
        partes.append(f"\n--- Rodada {i} ---")
        if turno.get("pergunta_ou_duvida"):
            partes.append(f"Pergunta/Dúvida: {turno['pergunta_ou_duvida']}")
        partes.append(f"Lição/Resposta: {turno['licao']}")
    return "\n".join(partes)


# --- Loop de Execução Principal ---
if __name__ == "__main__":
    print("=== BEM-VINDO À ESCOLA DE IA (conversa autônoma até esgotar o tema) ===")

    while True:
        tema = input("\nDigite um tema para a aula (ou 'sair'): ")
        if tema.lower() == 'sair':
            break

        conversa = []  # guarda todas as rodadas desse tema

        # ---------- RODADA 1: tentativa inicial do aluno ----------
        resposta_aluno = aluno_responde_inicial(tema)
        print(f"\n--- Resposta Inicial do Aluno (Local): ---\n{resposta_aluno}")

        feedback_1 = professor_1_avalia(tema, resposta_aluno)
        print(f"\n--- Feedback do Professor 1: ---\n{feedback_1}")

        feedback_3 = professor_3_segunda_opiniao(tema, resposta_aluno)
        print(f"\n--- Feedback do Professor 3: ---\n{feedback_3}")

        licao = professor_2_ensina(tema, resposta_aluno, feedback_1, feedback_3)
        print(f"\n--- Lição Ideal (Professor 2): ---\n{licao}")

        conversa.append({
            "pergunta_ou_duvida": tema,
            "licao": licao,
            "proxima_pergunta": None  # será preenchido se o aluno perguntar algo
        })

        # ---------- RODADAS SEGUINTES: dúvidas do aluno ----------
        for rodada in range(2, MAX_RODADAS_DUVIDA + 2):
            historico_texto = montar_historico_texto(tema, conversa)

            duvida = aluno_pergunta_duvida(tema, historico_texto)

            if PALAVRA_PARADA in duvida.upper():
                print(f"\n[Estudante Local] Não tenho mais dúvidas sobre '{tema}'. ✅")
                break

            print(f"\n--- Dúvida do Aluno: ---\n{duvida}")
            conversa[-1]["proxima_pergunta"] = duvida  # liga a dúvida à lição anterior

            feedback_1 = professor_1_responde_duvida(duvida, historico_texto)
            print(f"\n--- Resposta do Professor 1: ---\n{feedback_1}")

            feedback_3 = professor_3_responde_duvida(duvida, historico_texto)
            print(f"\n--- Resposta do Professor 3: ---\n{feedback_3}")

            licao = professor_2_responde_duvida(duvida, historico_texto, feedback_1, feedback_3)
            print(f"\n--- Resposta Final (Professor 2): ---\n{licao}")

            conversa.append({
                "pergunta_ou_duvida": duvida,
                "licao": licao,
                "proxima_pergunta": None
            })

            if rodada == MAX_RODADAS_DUVIDA + 1:
                print(f"\n[Sistema] Limite de {MAX_RODADAS_DUVIDA} rodadas de dúvida atingido. Encerrando o tema.")

        # ---------- Salva a conversa completa ----------
        salvar_conversa_treino(tema, conversa)

        print("\n" + "=" * 50)