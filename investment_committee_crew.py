# Databricks notebook source
# 1. Configuração e Imports
# O comando abaixo instala as bibliotecas necessárias no ambiente Databricks.
# Descomente as linhas abaixo se estiver executando em uma célula do notebook.
# %pip install crewai yfinance langchain-databricks duckduckgo-search matplotlib
# dbutils.library.restartPython()

import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from crewai import Agent, Task, Crew, Process
from langchain_databricks import ChatDatabricks
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# 2. Definição do LLM
# Lista de modelos disponíveis (Configuração do Usuário)
models = [
    'databricks-gpt-oss-20b',
    'databricks-gpt-oss-120b',
    'databricks-llama-4-maverick',
    'databricks-gemma-3-12b',
    'databricks-meta-llama-3-1-8b-instruct',
    'databricks-meta-llama-3-3-70b-instruct', # ESCOLHIDO PARA RACIOCÍNIO
    'databricks-gte-large-en',
    'databricks-meta-llama-3-1-405b-instruct'
]

# Instância do Modelo Principal
llm_databricks = ChatDatabricks(
    endpoint=models[5], # Usando Llama 3.3 70B para melhor raciocínio
    temperature=0.1,
    max_tokens=4000
)

# 3. Criação das Ferramentas (Custom Tools)

class StockAnalysisTools:
    @tool("fetch_stock_data")
    def fetch_stock_data(ticker: str):
        """
        Baixa dados históricos dos últimos 180 dias de um Ticker via yfinance.
        Calcula RSI (14), SMA_50 e SMA_200.
        Gera um gráfico de preço e médias móveis salvo em /tmp/analysis.png.
        Retorna um resumo string com preço atual e indicadores.
        """
        try:
            # Baixar dados
            stock = yf.Ticker(ticker)
            df = stock.history(period="180d")
            
            if df.empty:
                return f"Erro: Não foi possível encontrar dados para o ticker {ticker}."

            # Calcular Indicadores
            # SMA 50 e 200
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # RSI 14
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            current_price = df['Close'].iloc[-1]
            current_rsi = df['RSI'].iloc[-1]
            current_sma_50 = df['SMA_50'].iloc[-1]
            current_sma_200 = df['SMA_200'].iloc[-1]
            
            # Gerar Gráfico
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df['Close'], label='Preço de Fechamento', color='blue')
            plt.plot(df.index, df['SMA_50'], label='SMA 50', color='orange', linestyle='--')
            plt.plot(df.index, df['SMA_200'], label='SMA 200', color='red', linestyle='--')
            plt.title(f'Análise Técnica: {ticker}')
            plt.xlabel('Data')
            plt.ylabel('Preço')
            plt.legend()
            plt.grid(True)
            plt.savefig("/tmp/analysis.png")
            plt.close()
            
            summary = (
                f"Análise Técnica para {ticker}:\n"
                f"- Preço Atual: {current_price:.2f}\n"
                f"- RSI (14): {current_rsi:.2f}\n"
                f"- SMA (50): {current_sma_50:.2f}\n"
                f"- SMA (200): {current_sma_200:.2f}\n"
                f"Gráfico salvo em /tmp/analysis.png"
            )
            return summary
            
        except Exception as e:
            return f"Erro ao processar dados para {ticker}: {str(e)}"

    @tool("search_market_news")
    def search_market_news(ticker: str):
        """
        Busca as 5 manchetes/resumos mais recentes sobre o Ticker usando DuckDuckGo.
        Query: "{ticker} stock investment analysis news"
        """
        search = DuckDuckGoSearchRun()
        query = f"{ticker} stock investment analysis news"
        try:
            results = search.run(query)
            # O DuckDuckGoSearchRun retorna uma string única, vamos tentar limitar ou formatar se possível,
            # mas o comportamento padrão é retornar um resumo.
            return f"Notícias recentes para {ticker}:\n{results}"
        except Exception as e:
            return f"Erro ao buscar notícias para {ticker}: {str(e)}"

# 4. Definição dos Agentes (Personas)

# Agente 1: The Quant (Analista Técnico)
quant_agent = Agent(
    role='The Quant (Analista Técnico)',
    goal='Diagnosticar a saúde técnica do ativo (Tendência de Alta/Baixa, Suporte/Resistência).',
    backstory="""Você é um especialista em análise técnica quantitativa com anos de experiência em Wall Street. 
    Você analisa gráficos, médias móveis e indicadores como RSI para prever movimentos de preço. 
    Você é frio, calculista e baseia suas opiniões puramente em números e padrões gráficos.""",
    verbose=True,
    allow_delegation=False,
    llm=llm_databricks,
    tools=[StockAnalysisTools.fetch_stock_data]
)

# Agente 2: The Fundamentalist (Analista de Macro/Notícias)
fundamentalist_agent = Agent(
    role='The Fundamentalist (Analista de Macro/Notícias)',
    goal='Identificar catalisadores de notícias e sentimento do mercado (Fear vs Greed).',
    backstory="""Você é um analista fundamentalista sênior focado em notícias de mercado e macroeconomia. 
    Você entende como notícias, relatórios de ganhos e eventos globais impactam o preço das ações. 
    Você busca entender o 'porquê' por trás dos movimentos do mercado.""",
    verbose=True,
    allow_delegation=False,
    llm=llm_databricks,
    tools=[StockAnalysisTools.search_market_news]
)

# Agente 3: Portfolio Manager (Decisor)
portfolio_manager_agent = Agent(
    role='Portfolio Manager (Decisor)',
    goal='Sintetizar os relatórios e decidir a ação final (COMPRAR, VENDER, AGUARDAR) com Preço Alvo.',
    backstory="""Você é o chefe do comitê de investimentos. Sua responsabilidade é ouvir seus analistas 
    (Técnico e Fundamentalista), pesar as evidências e tomar a decisão final de investimento. 
    Você deve fornecer uma tese clara e um preço alvo justificado.""",
    verbose=True,
    allow_delegation=True, # Permitir delegação se necessário, embora o prompt diga "Apenas delegação" no sentido de orquestrar, mas tools=Nenhuma.
    llm=llm_databricks,
    tools=[] # Nenhuma ferramenta direta
)

# 5. Definição das Tasks

ticker_symbol = "PETR4.SA"

# Task 1: Análise Técnica
technical_analysis_task = Task(
    description=f"""
    Analise o ticker {ticker_symbol} usando a ferramenta fetch_stock_data.
    Identifique a tendência atual (alta, baixa, lateral).
    Analise o RSI e as Médias Móveis (SMA 50 e 200).
    Determine níveis de suporte e resistência se possível.
    Forneça um relatório técnico detalhado.
    """,
    expected_output="Um relatório de análise técnica detalhando tendência, RSI, SMAs e sinais de compra/venda técnicos.",
    agent=quant_agent
)

# Task 2: Análise Fundamentalista/Notícias
fundamental_analysis_task = Task(
    description=f"""
    Busque notícias recentes e sentimento de mercado para {ticker_symbol} usando a ferramenta search_market_news.
    Identifique fatos relevantes recentes, sentimento geral (positivo/negativo) e possíveis catalisadores.
    Resuma as principais manchetes e seu impacto potencial.
    """,
    expected_output="Um resumo das notícias mais recentes, sentimento do mercado e principais catalisadores para o ativo.",
    agent=fundamentalist_agent
)

# Task 3: Decisão de Investimento
investment_decision_task = Task(
    description=f"""
    Revise os relatórios de análise técnica e fundamentalista para {ticker_symbol}.
    Sintetize as informações.
    Decida a ação final: COMPRAR, VENDER ou AGUARDAR.
    Defina um Preço Alvo com base na análise.
    Gere um relatório final completo em Markdown justificando a decisão.
    """,
    expected_output="Um relatório final em Markdown contendo a Tese de Investimento, Ação Recomendada (Buy/Sell/Hold) e Preço Alvo.",
    agent=portfolio_manager_agent,
    context=[technical_analysis_task, fundamental_analysis_task] # Passar o contexto das tarefas anteriores
)

# 6. Execução

investment_crew = Crew(
    agents=[quant_agent, fundamentalist_agent, portfolio_manager_agent],
    tasks=[technical_analysis_task, fundamental_analysis_task, investment_decision_task],
    verbose=True,
    process=Process.sequential
)

print(f"Iniciando a análise para {ticker_symbol}...")
result = investment_crew.kickoff()

print("\n\n########################")
print("## RELATÓRIO FINAL ##")
print("########################\n")
print(result)

# Tentar exibir a imagem gerada (específico para Databricks)
try:
    from IPython.display import Image, display
    if os.path.exists("/tmp/analysis.png"):
        print("\nExibindo gráfico de análise técnica...")
        display(Image(filename="/tmp/analysis.png"))
    else:
        print("Imagem de análise não encontrada em /tmp/analysis.png")
except Exception as e:
    print(f"Não foi possível exibir a imagem: {e}")
