import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


# Conjunto de dados fictício: lista de e-mails com rótulos 'spam' ou 'não spam' (ham)
data = {
    'email': [
        "Compre agora a oferta grátis",  # SPAM
        "Clique aqui para ganhar prêmio",  # SPAM
        "Reunião agendada para segunda",  # NÃO SPAM
        "Oferta exclusiva para você",  # SPAM
        "Vamos almoçar amanhã?",  # NÃO SPAM
        "Grátis: sua consulta médica",  # SPAM
        "Confirme sua inscrição no evento",  # NÃO SPAM
        "Grátis, clique aqui para baixar",  # SPAM
        "Atualização importante do projeto",  # NÃO SPAM
        "Oferta limitada, clique e ganhe",  # SPAM
        "Seu extrato bancário está disponível",  # NÃO SPAM
        "Ganhe dinheiro rapidamente, clique aqui",  # SPAM
        "Agenda de viagem confirmada",  # NÃO SPAM
        "Você foi selecionado! Grátis!",  # SPAM
        "Temos um convite especial para você",  # SPAM
        "Relatório financeiro entregue",  # NÃO SPAM
        "Receba prêmios grátis agora mesmo",  # SPAM
        "E-mail com novidades do trabalho",  # NÃO SPAM
        "Oferta incrível, não perca!",  # SPAM
        "Que tal nos encontrarmos hoje?",  # NÃO SPAM
        "Receba cupons grátis aqui",  # SPAM
        "Participe da reunião amanhã",  # NÃO SPAM
        "Aproveite esta promoção especial",  # SPAM
        "Confirmando o horário do encontro",  # NÃO SPAM
        "Parabéns, você ganhou um sorteio!",  # SPAM
        "Seu pedido foi enviado",  # NÃO SPAM
        "Clique para desbloquear sua oferta",  # SPAM
        "Segue o documento solicitado",  # NÃO SPAM
        "Compre com desconto imperdível",  # SPAM
        "Lembrete: reunião amanhã às 10h",  # NÃO SPAM
        "Grátis, sua inscrição foi aprovada",  # SPAM
        "Convite para o workshop de amanhã",  # NÃO SPAM
        "Ganhe prêmios incríveis hoje mesmo!",  # SPAM
        "Solicitação de reunião confirmada",  # NÃO SPAM
        "Oferta especial por tempo limitado",  # SPAM
        "Atualização semanal do time",  # NÃO SPAM
        "Parabéns, você ganhou um presente!",  # SPAM
        "Resumo das atividades do dia",  # NÃO SPAM
        "Clique aqui para obter seu desconto",  # SPAM
        "Envio confirmado para amanhã",  # NÃO SPAM
        "Oferta grátis, veja os detalhes",  # SPAM
        "Combinamos os detalhes do projeto",  # NÃO SPAM
        "Ganhe recompensas incríveis agora",  # SPAM
        "Confirme o agendamento de reunião",  # NÃO SPAM
        "Sua conta precisa de atenção urgente",  # NÃO SPAM
        "Promoção imperdível apenas hoje",  # SPAM
        "Seu relatório está disponível",  # NÃO SPAM
        "Oferta relâmpago para você",  # SPAM
        "Podemos confirmar o pedido?",  # NÃO SPAM
        "Atualização: tarefas da equipe",  # NÃO SPAM
        "Você foi selecionado, clique agora!",  # SPAM
        "Sua encomenda foi entregue",  # NÃO SPAM,
        "Aproveite esta chance única",  # SPAM
        "Seu documento está anexado",  # NÃO SPAM,
        "Ganhe um presente exclusivo, grátis!",  # SPAM
        "Planejamento semanal enviado",  # NÃO SPAM,
        "Compre 2 e leve 3 agora!",  # SPAM
        "Lembrete de reunião com o cliente",  # NÃO SPAM,
        "Baixe seu voucher de desconto aqui",  # SPAM,
        "Resumo do mês está disponível",  # NÃO SPAM,
        "Oferta VIP exclusiva para você",  # SPAM,
        "Reunião confirmada às 15h",  # NÃO SPAM,
        "Receba agora mesmo seus prêmios!",  # SPAM,
        "Seu boleto está disponível",  # NÃO SPAM,
        "Oferta grátis imperdível hoje!",  # SPAM,
        "Email com atualização semanal",  # NÃO SPAM,
        "Clique para ganhar um bônus especial",  # SPAM,
        "Relatório do projeto concluído",  # NÃO SPAM,
        "Desconto exclusivo, clique agora!",  # SPAM,
        "Proposta de contrato revisada",  # NÃO SPAM,
        "Participe da nossa pesquisa",  # NÃO SPAM,
        "Ganhe dinheiro fácil hoje mesmo",  # SPAM,
        "Reunião online confirmada",  # NÃO SPAM,
        "Grátis, acesse já sua recompensa",  # SPAM,
        "Aviso de manutenção no sistema",  # NÃO SPAM,
        "Clique para confirmar sua inscrição",  # SPAM,
        "Resumo das entregas do projeto",  # NÃO SPAM,
        "Aproveite esta oportunidade grátis",  # SPAM,
        "Detalhes da reunião em anexo",  # NÃO SPAM,
        "Ganhe mais com nossas promoções",  # SPAM,
        "Email com novidades importantes",  # NÃO SPAM
    ],
    'label': [
        "spam", "spam", "não spam", "spam", "não spam",
        "spam", "não spam", "spam", "não spam", "spam",
        "não spam", "spam", "não spam", "spam", "spam",
        "não spam", "spam", "não spam", "spam", "não spam",
        "spam", "não spam", "spam", "não spam", "spam",
        "não spam", "spam", "não spam", "spam", "não spam",
        "spam", "não spam", "spam", "não spam", "spam",
        "não spam", "spam", "não spam", "spam", "não spam",
        "spam", "não spam", "não spam", "spam", "não spam",
        "spam", "não spam", "spam", "não spam", "não spam",
        "spam", "não spam", "spam", "não spam", "spam",
        "não spam", "spam", "não spam", "spam", "não spam",
        "spam", "não spam", "spam", "não spam", "não spam",
        "spam", "não spam", "spam", "não spam", "não spam",
        "spam", "não spam", "spam", "não spam", "não spam",
        "spam", "não spam", "spam", "não spam", "não spam",
        "spam"
    ]
}

df = pd.DataFrame(data)

df.head()

x = df['email']
y = df['label']

vetorizar = CountVectorizer()
X_vetorized = vetorizar.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X_vetorized, y, test_size=0.3, random_state=42)

naive_bayes = MultinomialNB()

naive_bayes.fit(X_train, y_train)


y_pred = naive_bayes.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("A acurácia é: ", accuracy)


novo_email = ['sua compra foi aprovada, clique para confirmar']
novo_email_vetorizado = vetorizar.transform(novo_email)

previsao = naive_bayes.predict(novo_email_vetorizado)
print('Predição do novo e-mail é: ', previsao)

