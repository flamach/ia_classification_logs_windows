# Script PowerShell pour démarrer la stack de monitoring Prometheus + Grafana
# Usage: .\start_monitoring.ps1

Write-Host "=== DÉMARRAGE STACK MONITORING XGBOOST ===" -ForegroundColor Green

# Vérifier Docker
Write-Host "Vérification Docker..." -ForegroundColor Yellow
try {
    docker --version | Out-Null
    Write-Host "Docker détecté ✓" -ForegroundColor Green
} catch {
    Write-Host "Docker non trouvé. Installez Docker Desktop et relancez." -ForegroundColor Red
    exit 1
}

# Démarrer la stack
Write-Host "Démarrage des conteneurs..." -ForegroundColor Yellow
docker-compose up -d

# Attendre que les services soient prêts
Write-Host "Attente démarrage des services..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Vérifier les services
Write-Host "Vérification des services..." -ForegroundColor Yellow

$services = @(
    @{Name="Prometheus"; URL="http://localhost:9090"; Port=9090},
    @{Name="Pushgateway"; URL="http://localhost:9091"; Port=9091},
    @{Name="Grafana"; URL="http://localhost:3000"; Port=3000}
)

foreach ($service in $services) {
    try {
        $response = Invoke-WebRequest -Uri $service.URL -TimeoutSec 5 -UseBasicParsing
        Write-Host "$($service.Name) ✓ - $($service.URL)" -ForegroundColor Green
    } catch {
        Write-Host "$($service.Name) ✗ - Erreur connexion port $($service.Port)" -ForegroundColor Red
    }
}

Write-Host "`n=== INFORMATIONS DE CONNEXION ===" -ForegroundColor Cyan
Write-Host "Prometheus: http://localhost:9090" -ForegroundColor White
Write-Host "Pushgateway: http://localhost:9091" -ForegroundColor White
Write-Host "Grafana: http://localhost:3000" -ForegroundColor White
Write-Host "  - Utilisateur: admin" -ForegroundColor Gray
Write-Host "  - Mot de passe: admin123" -ForegroundColor Gray

Write-Host "`n=== COMMANDES UTILES ===" -ForegroundColor Cyan
Write-Host "Voir les logs: docker-compose logs -f" -ForegroundColor White
Write-Host "Arrêter: docker-compose down" -ForegroundColor White
Write-Host "Redémarrer: docker-compose restart" -ForegroundColor White

Write-Host "`n=== PROCHAINES ÉTAPES ===" -ForegroundColor Cyan
Write-Host "1. Connectez-vous à Grafana (admin/admin123)" -ForegroundColor White
Write-Host "2. Le dashboard XGBoost sera automatiquement provisionné" -ForegroundColor White
Write-Host "3. Lancez l'entraînement: python train_model_xgboost_monitored.py" -ForegroundColor White
Write-Host "4. Consultez les métriques en temps réel sur le dashboard" -ForegroundColor White

Write-Host "`nStack de monitoring démarrée avec succès ! 🚀" -ForegroundColor Green
